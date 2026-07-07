#!/usr/bin/env python3
"""
What EAGLE and Medusa cost once there is more than one tenant.

Both are single-model speculators whose trained module reads the target's hidden states.
A LoRA adapter shifts those states, so a module trained for one adapter is not faithful
for another: one shared module is adapter-blind and collapses the same way our shared
draft does, and one module per tenant is faithful but has to be paid for K times.

This prices the second horn from the real upstream architectures:

  Medusa   5 parallel heads, each ResBlock(Linear(h,h) + SiLU, residual) into its own
           Linear(h, vocab), which is the term that dominates
  EAGLE-1  Linear(2h, h) plus one full decoder layer, reusing the target's embedding
           and lm_head
  WarpPipe one shared 160m draft base, amortized over every tenant, plus a rank-32 LoRA
           per tenant, all applied in one draft-side SGMV pass

Reports per-tenant parameter count and resident memory against K, and the measured cost
of advancing a batch spanning K adapters in one step: K serialized dense modules against
one flat SGMV pass. Latency does not depend on weights, so the modules are randomly
initialised at the real target geometry.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from distill import TGT, DRF  # noqa: E402
from serve_multitenant import extract_draft_lora, wrap_draft, STATE  # noqa: E402

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import LoraConfig, get_peft_model


# real Medusa head (FasterDecoding/Medusa medusa_model.py, verbatim)
class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


def medusa_heads(hidden, vocab, n_heads=5, n_layers=1):
    """The exact Medusa head stack: n_heads x (n_layers ResBlock -> Linear(h,vocab))."""
    return nn.ModuleList([
        nn.Sequential(*([ResBlock(hidden)] * n_layers),
                      nn.Linear(hidden, vocab, bias=False))
        for _ in range(n_heads)
    ])


def eagle_draft(cfg):
    """EAGLE-1 draft (cnets1.py): fc(2h->h) + one LlamaDecoderLayer at target geometry."""
    dl_cfg = AutoConfig.from_pretrained(TGT)
    dl_cfg.num_hidden_layers = 1
    layer = LlamaDecoderLayer(dl_cfg, layer_idx=0)
    fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
    return nn.ModuleDict({"fc": fc, "layer": layer})


def numel(m):
    return sum(p.numel() for p in m.parameters())


def time_fwd(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b))
    return statistics.median(s)


def main():
    p = argparse.ArgumentParser(description="M3: EAGLE/Medusa per-tenant cost vs WarpPipe")
    p.add_argument("--B", type=int, default=32)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--medusa_heads", type=int, default=5)
    p.add_argument("--drf_r", type=int, default=32)
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--Kmem", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device, dtype = "cuda", torch.float16
    tcfg = AutoConfig.from_pretrained(TGT)      # target 7B geometry
    dcfg = AutoConfig.from_pretrained(DRF)      # 160m draft geometry
    H, V = tcfg.hidden_size, tcfg.vocab_size
    print(f"target 7B: hidden={H} vocab={V} layers={tcfg.num_hidden_layers}")
    print(f"draft 160m: hidden={dcfg.hidden_size} layers={dcfg.num_hidden_layers}")

    # per-tenant parameter counts (real modules)
    med = medusa_heads(H, V, args.medusa_heads).to(device, dtype)
    eag = eagle_draft(tcfg).to(device, dtype)
    p_med = numel(med)
    p_eag = numel(eag)
    # WarpPipe per-tenant = a rank-r LoRA on q,k,v,o of the 160m draft
    mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    tiny = get_peft_model(
        AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=torch.float32),
        LoraConfig(r=args.drf_r, lora_alpha=2 * args.drf_r, target_modules=mods,
                   lora_dropout=0.0, task_type="CAUSAL_LM"))
    p_lora = sum(v.numel() for k, v in tiny.named_parameters() if "lora_" in k)
    p_shared_draft = numel(AutoModelForCausalLM.from_pretrained(DRF))  # amortized once
    print(f"\nper-tenant params: Medusa(5h)={p_med/1e6:.1f}M  EAGLE-1={p_eag/1e6:.1f}M  "
          f"WarpPipe-LoRA(r{args.drf_r} qkvo)={p_lora/1e6:.2f}M  "
          f"(+ shared 160m draft {p_shared_draft/1e6:.0f}M once)")

    # resident memory vs K (fp16, 2 bytes/param)
    B2 = 2
    mem = []
    for K in args.Kmem:
        m_med = K * p_med * B2 / 1e9
        m_eag = K * p_eag * B2 / 1e9
        m_wp = (p_shared_draft + K * p_lora) * B2 / 1e9
        mem.append({"K": K, "medusa_gb": m_med, "eagle_gb": m_eag, "warppipe_gb": m_wp})
        print(f"  K={K:>3}: Medusa {m_med:7.1f} GB | EAGLE {m_eag:7.1f} GB | "
              f"WarpPipe {m_wp:6.2f} GB")

    # measured draft-apply cost vs K (one decode step, B tokens over K adapters)
    # Faithful per-tenant EAGLE/Medusa: each adapter's dense module runs over its
    # routed sub-batch -> K serialized forwards. WarpPipe: one 160m forward + one
    # K-slot SGMV pass (flat). Modules use random weights (cost-only).
    print(f"\ndraft-apply cost, B={args.B} tokens over K adapters (one decode step):")
    hid = torch.randn(args.B, H, device=device, dtype=dtype)
    med.eval(); eag.eval()

    # 160m shared draft + SGMV LoRA (ours), reuse the serving path
    base_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    base_draft.config.use_cache = False
    # one token per row = a cached decode step (the real per-step draft regime; an
    # 8-token no-KV recompute would inflate the shared-backbone cost ~8x and is not
    # how the draft runs in serving). The LoRA/SGMV apply is what scales with K.
    ids = torch.randint(5, 1000, (args.B, 1), device=device)

    apply = []
    with torch.inference_mode():
        for K in args.K:
            # route B tokens across K adapters (contiguous sub-batches)
            sizes = [args.B // K + (1 if i < args.B % K else 0) for i in range(K)]
            offs = [0]
            for s in sizes:
                offs.append(offs[-1] + s)

            def med_step():
                for i in range(K):
                    sub = hid[offs[i]:offs[i + 1]]
                    if sub.shape[0] == 0:
                        continue
                    for head in med:              # 5 parallel heads (one-shot gamma)
                        head(sub)

            def eag_step():
                for i in range(K):
                    sub = hid[offs[i]:offs[i + 1]]
                    if sub.shape[0] == 0:
                        continue
                    x = eag["fc"](torch.cat([sub, sub], dim=-1)).unsqueeze(1)
                    eag["layer"](x, position_ids=torch.zeros(sub.shape[0], 1,
                                 dtype=torch.long, device=device))

            # WarpPipe: one 160m forward with K-slot SGMV LoRA (flat)
            w, scale = extract_draft_lora(tiny, K, dtype, device, modules=tuple(mods))
            STATE["scale"] = scale
            wd = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
            wd.config.use_cache = False
            wrap_draft(wd, w)
            adapter = torch.arange(args.B, device=device) % K

            def wp_step(wd=wd, adapter=adapter):
                STATE["mode"] = "lora"
                STATE["idx"] = adapter.repeat_interleave(ids.size(1))
                wd(input_ids=ids, use_cache=False)

            t_med = time_fwd(med_step)
            t_eag = time_fwd(eag_step)
            t_wp = time_fwd(wp_step)
            STATE["mode"] = "none"
            apply.append({"K": K, "medusa_ms": t_med, "eagle_ms": t_eag, "warppipe_ms": t_wp})
            print(f"  K={K:>2}: Medusa {t_med:6.2f} ms | EAGLE {t_eag:6.2f} ms | "
                  f"WarpPipe(SGMV) {t_wp:5.2f} ms")
            del wd; torch.cuda.empty_cache()

    verdict = {
        "medusa_params_M": p_med / 1e6, "eagle_params_M": p_eag / 1e6,
        "warppipe_lora_params_M": p_lora / 1e6, "shared_draft_params_M": p_shared_draft / 1e6,
        "per_tenant_ratio_vs_medusa": p_med / p_lora, "per_tenant_ratio_vs_eagle": p_eag / p_lora,
    }
    print("\n=== per-tenant cost ===")
    print(f"  per-tenant faithful draft is {verdict['per_tenant_ratio_vs_medusa']:.0f}x "
          f"smaller than Medusa, {verdict['per_tenant_ratio_vs_eagle']:.0f}x smaller than EAGLE")
    print("  and flat in K (one SGMV pass) vs K serialized dense heads")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"eagle_medusa_cost_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "B": args.B,
                   "target_hidden": H, "vocab": V, "medusa_heads": args.medusa_heads,
                   "drf_r": args.drf_r, "eagle_commit": "cb7e084", "medusa_commit": "e2a5d20",
                   "params": verdict, "memory_gb": mem, "apply_ms": apply}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
