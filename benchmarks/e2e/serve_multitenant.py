#!/usr/bin/env python3
"""
Multi-tenant speculative serving: K co-resident tenants, one draft pass.

Runs B concurrent requests, each routed to one of K adapters, through four draft
strategies and reports throughput and draft-step cost against K:

  no_spec       plain batched target decode, the floor every tier is compared to
  shared_draft  one adapter-blind draft for all tenants: cheap, but low acceptance
  per_adapter   each tenant's own draft-LoRA, applied in K separate draft forwards,
                which is what a single-model spec engine has to do
  sgmv_draft    the same per-tenant draft-LoRAs applied to the whole batch in one
                draft-side SGMV pass

per_adapter and sgmv_draft use identical weights and therefore have identical
acceptance, which is what isolates the cost claim: the only thing that differs is how
the K adapters reach the draft forward. Tenants share a task here so acceptance is
held fixed across K; heterogeneous traffic is served in serve_mixed_traffic.py and
distinct per-tenant drafts are exercised in multitenant_matrix.py.

Decode here is full-recompute with no KV cache. That is fine for the COST sweep this
script exists for (t_sgmv vs t_per_adapter against K, where every strategy pays the same
target) but it must not be read as a serving speedup. Recomputing the prefix makes one
decoded token cost about what verifying gamma+1 of them costs, which flatters any
speculative tier; with a KV cache the target step collapses and an eagerly-run draft costs
more than the decode it replaces. End-to-end serving numbers therefore come from
serve_mixed_traffic.py, which is KV-cached and graph-captured on every tier.

Also exports the draft-side SGMV plumbing the rest of the suite uses:
`extract_draft_lora` stacks K per-adapter LoRAs into BGMV slot tensors, `wrap_draft`
routes a draft model's linears through them, and `STATE` carries the per-row adapter
index into the wrapped forward.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from distill import train_lora, CORPUS, TGT, DRF  # noqa: E402

from vllm.lora.ops.bgmv_shrink import bgmv_shrink  # noqa: E402
from vllm.lora.ops.bgmv_expand import bgmv_expand  # noqa: E402

# Shared state read by every wrapped draft linear during a forward.
STATE = {"mode": "none", "idx": None, "scale": 2.0}


class BGMVLoRALinear(nn.Module):
    """Base nn.Linear + per-row draft-LoRA delta via vLLM BGMV, K adapter slots.

    mode 'none'  -> base only (shared_draft / unfaithful).
    mode 'lora'  -> base + BGMV delta with STATE['idx'] (one pass, any routing);
                    sgmv_draft passes per-row adapter ids, per_adapter passes a
                    constant id over its sub-batch.
    """

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.A = A  # [K, rank, in]
        self.B = B  # [K, out, rank]
        self.rank = A.size(1)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if STATE["mode"] == "none":
            return base
        idx = STATE["idx"]
        shp = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()
        n = x2d.size(0)
        h = torch.zeros(n, self.rank, device=x.device, dtype=x.dtype)
        delta = torch.zeros(n, self.out_features, device=x.device, dtype=x.dtype)
        bgmv_shrink(x2d, self.A, h, idx, STATE["scale"])
        bgmv_expand(h, self.B, delta, idx, True)
        return base + delta.reshape(*shp[:-1], self.out_features)


def extract_draft_lora(peft_draft, K, dtype, device,
                       modules=("q_proj", "v_proj")):
    """Pull lora_A/lora_B for the given attention projections from the distilled
    PEFT draft into BGMV-layout tensors, replicated across K adapter slots (all
    tenants share the task, so slots are identical, though the routing/BGMV work is
    still real). `modules` must match what the draft was distilled on, so every
    trained LoRA delta is applied on the draft path (else t_sgmv would omit real
    work and understate the WarpPipe draft cost)."""
    weights, scale = {}, 2.0
    for name, mod in peft_draft.named_modules():
        if hasattr(mod, "lora_A") and any(name.endswith(m) for m in modules):
            # base_model.model.model.layers.{l}.self_attn.{q,v}_proj -> suffix key
            key = name.split("base_model.model.")[-1]
            A = mod.lora_A["default"].weight.detach()          # [rank, in]
            B = mod.lora_B["default"].weight.detach()          # [out, rank]
            scale = float(mod.scaling["default"])
            Ak = A.unsqueeze(0).repeat(K, 1, 1).to(device, dtype).contiguous()
            Bk = B.unsqueeze(0).repeat(K, 1, 1).to(device, dtype).contiguous()
            weights[key] = (Ak, Bk)
    return weights, scale


def wrap_draft(base_draft, weights):
    """Replace q_proj/v_proj on the base draft with BGMV-LoRA wrappers."""
    n = 0
    for name, mod in list(base_draft.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if full in weights and isinstance(child, nn.Linear):
                A, B = weights[full]
                setattr(mod, cname, BGMVLoRALinear(child, A, B))
                n += 1
    return n


def pad_batch(rows, pad_id, device):
    """Left-pad a list of 1D id tensors -> ids[B,S], mask[B,S], pos[B,S].
    Left padding keeps the last position real for every row, so next-token
    logits and end-relative indexing are uniform across ragged rows."""
    S = max(r.numel() for r in rows)
    B = len(rows)
    ids = torch.full((B, S), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros(B, S, device=device, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, S - r.numel():] = r
        mask[i, S - r.numel():] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


@torch.inference_mode()
def draft_step(draft, rows, adapter, strategy, pad_id, device):
    """One draft next-token for every row, per strategy. Returns nt[B]."""
    B = len(rows)
    if strategy == "shared_draft":
        STATE["mode"] = "none"
        ids, mask, pos = pad_batch(rows, pad_id, device)
        logits = draft(input_ids=ids, attention_mask=mask, position_ids=pos,
                       use_cache=False).logits[:, -1]
        return logits.argmax(-1)
    if strategy == "sgmv_draft":
        STATE["mode"] = "lora"
        ids, mask, pos = pad_batch(rows, pad_id, device)
        S = ids.size(1)
        STATE["idx"] = adapter.repeat_interleave(S)      # per token-row
        logits = draft(input_ids=ids, attention_mask=mask, position_ids=pos,
                       use_cache=False).logits[:, -1]
        return logits.argmax(-1)
    if strategy == "per_adapter":
        # K separate draft forwards over per-adapter sub-batches (serialized).
        nt = torch.zeros(B, device=device, dtype=torch.long)
        STATE["mode"] = "lora"
        for a in adapter.unique().tolist():
            sub = (adapter == a).nonzero(as_tuple=True)[0]
            subrows = [rows[i] for i in sub.tolist()]
            ids, mask, pos = pad_batch(subrows, pad_id, device)
            S = ids.size(1)
            STATE["idx"] = torch.full((ids.numel(),), a, device=device, dtype=torch.long)
            logits = draft(input_ids=ids, attention_mask=mask, position_ids=pos,
                           use_cache=False).logits[:, -1]
            nt[sub] = logits.argmax(-1)
        return nt
    raise ValueError(strategy)


@torch.inference_mode()
def spec_iter(draft, target, rows, adapter, gamma, strategy, pad_id, device):
    """One speculative outer iteration over the whole batch. Returns
    (new_rows, produced_tokens) where produced = sum(accepted+1 bonus)."""
    B = len(rows)
    # 1) draft proposes gamma tokens/row
    work = [r.clone() for r in rows]
    proposed = [[] for _ in range(B)]
    for _ in range(gamma):
        nt = draft_step(draft, work, adapter, strategy, pad_id, device)
        for i in range(B):
            proposed[i].append(nt[i])
            work[i] = torch.cat([work[i], nt[i].view(1)])
    # 2) target verify (single batched forward; target LoRA already applied)
    STATE["mode"] = "none"
    ids, mask, pos = pad_batch(work, pad_id, device)
    tgt_arg = target(input_ids=ids, attention_mask=mask, position_ids=pos,
                     use_cache=False).logits.argmax(-1)   # [B,S]
    new_rows, produced = [], 0
    for i in range(B):
        a = 0
        for j in range(gamma):
            # tgt prediction for proposed[j] sits at end-index -(gamma+1)+j
            if tgt_arg[i, -(gamma + 1) + j].item() == proposed[i][j].item():
                a += 1
            else:
                break
        bonus = tgt_arg[i, -(gamma + 1) + a]
        acc = torch.stack(proposed[i][:a]) if a else rows[i].new_empty(0)
        new_rows.append(torch.cat([rows[i], acc, bonus.view(1)]))
        produced += a + 1
    return new_rows, produced


@torch.inference_mode()
def nospec_iter(target, rows, pad_id, device):
    """One batched target autoregressive step: 1 token/row."""
    STATE["mode"] = "none"
    ids, mask, pos = pad_batch(rows, pad_id, device)
    nt = target(input_ids=ids, attention_mask=mask, position_ids=pos,
                use_cache=False).logits[:, -1].argmax(-1)
    return [torch.cat([rows[i], nt[i].view(1)]) for i in range(len(rows))], len(rows)


def draft_cost_ms(draft, B, K, adapter, strategy, pad_id, device, iters=30, warmup=8):
    """Isolated one-step draft cost (the crux: flat in K for sgmv, rising for
    per_adapter)."""
    rows = [torch.randint(5, 1000, (12,), device=device) for _ in range(B)]
    for _ in range(warmup):
        draft_step(draft, rows, adapter, strategy, pad_id, device)
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); draft_step(draft, rows, adapter, strategy, pad_id, device); b.record()
        b.synchronize(); s.append(a.elapsed_time(b))
    return statistics.median(sorted(s))


def run_strategy(draft, target, prompts_ids, adapter, gamma, strategy, iters,
                 pad_id, device):
    rows = [p.clone() for p in prompts_ids]
    # warmup one iter (JIT/config caches), not timed
    if strategy == "no_spec":
        rows, _ = nospec_iter(target, rows, pad_id, device)
    else:
        rows, _ = spec_iter(draft, target, rows, adapter, gamma, strategy, pad_id, device)
    rows = [p.clone() for p in prompts_ids]
    torch.cuda.synchronize()
    t0 = time.time(); produced = 0
    for _ in range(iters):
        if strategy == "no_spec":
            rows, p = nospec_iter(target, rows, pad_id, device)
        else:
            rows, p = spec_iter(draft, target, rows, adapter, gamma, strategy, pad_id, device)
        produced += p
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {"produced": produced, "sec": dt, "tok_s": produced / dt,
            "tok_per_iter": produced / iters}


def main():
    p = argparse.ArgumentParser(description="multi-tenant speculative serving")
    p.add_argument("--tgt_steps", type=int, default=120)
    p.add_argument("--drf_steps", type=int, default=300)
    p.add_argument("--requests", type=int, default=32, help="B concurrent requests")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--iters", type=int, default=8, help="timed outer iterations")
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--drf_r", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=96)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device, dtype = "cuda", torch.float16
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(TGT)
    pad_id = tok.eos_token_id or 0
    task_text = CORPUS.upper()

    print("Stage 1: train target LoRA (llama-7b, UPPERCASE)")
    target = train_lora(TGT, tok, task_text, args.r, args.tgt_steps, 2e-4,
                        args.seqlen, batch=2, device=device, grad_ckpt=True, tag="tgt")
    target.gradient_checkpointing_disable()
    target.config.use_cache = False
    target.eval()

    print("Stage 2: distill draft LoRA (llama-160m, fp32)")
    peft_draft = train_lora(DRF, tok, task_text, args.drf_r, args.drf_steps, 2e-4,
                            args.seqlen, batch=8, device=device, grad_ckpt=False,
                            tag="drf", dtype=torch.float32)

    print("Stage 3: build BGMV-LoRA draft (extract distilled weights, K slots)")
    base_draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
    base_draft.config.use_cache = False

    # fixed multi-tenant prompt batch (uppercase task)
    base_prompts = ["THE HISTORY OF SCIENCE IS", "MACHINE LEARNING ENABLES",
                    "MODERN PHYSICS DESCRIBES", "COMPUTERS PROCESS INFORMATION",
                    "THE INTERNET CONNECTS", "BIOLOGY EXAMINES LIVING",
                    "CHEMISTRY STUDIES MATTER", "ECONOMICS ANALYZES HOW"]
    prompts_ids = [tok(base_prompts[i % len(base_prompts)],
                       return_tensors="pt").input_ids[0].to(device)
                   for i in range(args.requests)]

    all_rows, draft_cost = [], []
    for K in args.K:
        weights, scale = extract_draft_lora(peft_draft, K, dtype, device)
        STATE["scale"] = scale
        # fresh wrap each K (new slot count)
        draft = AutoModelForCausalLM.from_pretrained(DRF, torch_dtype=dtype).to(device).eval()
        draft.config.use_cache = False
        nwrap = wrap_draft(draft, weights)
        adapter = torch.arange(args.requests, device=device) % K

        print(f"\n=== K={K}  (wrapped {nwrap} draft linears) ===")
        # crux: isolated draft-step cost per strategy
        for strat in ("shared_draft", "sgmv_draft", "per_adapter"):
            ms = draft_cost_ms(draft, args.requests, K, adapter, strat, pad_id, device)
            draft_cost.append({"K": K, "strategy": strat, "draft_step_ms": ms})

        # end-to-end throughput per strategy
        for strat in ("no_spec", "shared_draft", "per_adapter", "sgmv_draft"):
            res = run_strategy(draft, target, prompts_ids, adapter, args.gamma,
                               strat, args.iters, pad_id, device)
            row = {"K": K, "strategy": strat, **res}
            all_rows.append(row)
            print(f"  {strat:>13}: {res['tok_s']:8.1f} tok/s   "
                  f"{res['tok_per_iter']:5.2f} tok/iter   ({res['sec']:.2f}s)")
        del draft
        torch.cuda.empty_cache()

    # verdicts
    def tps(K, s):
        return next(r["tok_s"] for r in all_rows if r["K"] == K and r["strategy"] == s)
    def dcost(K, s):
        return next(r["draft_step_ms"] for r in draft_cost if r["K"] == K and r["strategy"] == s)

    Kmax = max(args.K)
    sgmv_beats_shared = tps(Kmax, "sgmv_draft") > tps(Kmax, "shared_draft")
    sgmv_beats_peradp = tps(Kmax, "sgmv_draft") >= tps(Kmax, "per_adapter")
    sgmv_beats_nospec = all(tps(K, "sgmv_draft") > tps(K, "no_spec") for K in args.K)
    # draft-cost flatness: sgmv grows < 1.5x from min-K to max-K; per_adapter grows more
    sgmv_ratio = dcost(Kmax, "sgmv_draft") / dcost(min(args.K), "sgmv_draft")
    peradp_ratio = dcost(Kmax, "per_adapter") / dcost(min(args.K), "per_adapter")
    flat = sgmv_ratio < peradp_ratio and sgmv_ratio < 1.8

    verdict = "PASS" if (sgmv_beats_shared and sgmv_beats_peradp and
                         sgmv_beats_nospec and flat) else "PARTIAL"
    print("\n=== summary ===")
    print(f"  sgmv > shared (acceptance) @K={Kmax}: {sgmv_beats_shared} "
          f"({tps(Kmax,'sgmv_draft'):.0f} vs {tps(Kmax,'shared_draft'):.0f} tok/s)")
    print(f"  sgmv >= per_adapter (cost) @K={Kmax}: {sgmv_beats_peradp} "
          f"({tps(Kmax,'sgmv_draft'):.0f} vs {tps(Kmax,'per_adapter'):.0f} tok/s)")
    print(f"  sgmv > no_spec (net win) all K: {sgmv_beats_nospec}")
    print(f"  draft-step flat in K: sgmv x{sgmv_ratio:.2f} vs per_adapter x{peradp_ratio:.2f} -> {flat}")
    print(f"  {verdict}")

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"serve_multitenant_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "requests": args.requests,
                   "gamma": args.gamma, "iters": args.iters, "K": args.K,
                   "throughput": all_rows, "draft_cost": draft_cost,
                   "verdict": verdict}, f, indent=2)
    print("written:", jpath)


if __name__ == "__main__":
    main()
