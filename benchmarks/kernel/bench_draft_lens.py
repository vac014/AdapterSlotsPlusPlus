#!/usr/bin/env python3
"""
Draft-side SGMV inside a real transformer step, on stock vLLM kernels.

The microbenchmark shows the op is flat in K on its own. This asks whether the same primitive
survives being dropped into a real draft-model forward, using vLLM's own BGMV kernels with no
AdapterSlots code, and still keeps multi-tenant drafting flat in K at the model-step level.

A real OPT-125m decoder (built from its config, random weights, since this is a latency and
integration claim) with every decoder linear wrapped to add a per-request LoRA delta through
bgmv_shrink/expand. B draft rows, each routed to one of K adapters, in three modes: no LoRA,
one bgmv pair per linear over the whole batch, and one call per adapter per linear.

Reports draft-step latency against K, the LoRA overhead over the base forward, and the batched
against sequential speedup. The batched delta is checked against an fp32 reference in one layer.

Context for why this matters: stock vLLM's speculative proposer is LoraNotSupportedWorkerBase,
so the draft path forbids LoRA outright. The missing capability is a cheap, portable addition
to a real model step.
"""

import argparse
import csv
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM

from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.bgmv_expand import bgmv_expand

# Shared state read by every wrapped linear during a forward.
STATE = {"mode": "none", "idx": None, "scale": 1.0}


class LoRAWrappedLinear(nn.Module):
    """Wraps a base nn.Linear, adds a per-request LoRA delta via vLLM BGMV."""

    def __init__(self, base: nn.Linear, K: int, rank: int, dtype, device):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        g = torch.Generator(device=device).manual_seed(hash(id(self)) & 0xFFFF)
        self.A = (torch.randn(K, rank, self.in_features, device=device,
                              dtype=dtype, generator=g) * 0.02).contiguous()
        self.B = (torch.randn(K, self.out_features, rank, device=device,
                              dtype=dtype, generator=g) * 0.02).contiguous()
        self.rank = rank
        self.K = K

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        mode = STATE["mode"]
        if mode == "none":
            return base
        idx = STATE["idx"]
        scale = STATE["scale"]
        shp = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()
        n = x2d.size(0)
        delta = torch.zeros(n, self.out_features, device=x.device, dtype=x.dtype)
        hbuf = torch.zeros(n, self.rank, device=x.device, dtype=x.dtype)
        if mode == "batched":
            bgmv_shrink(x2d, self.A, hbuf, idx, scale)
            bgmv_expand(hbuf, self.B, delta, idx, True)
        elif mode == "seq_bgmv":
            for a in range(self.K):
                rows = (idx == a).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                xa = x2d.index_select(0, rows).contiguous()
                ha = torch.zeros(rows.numel(), self.rank, device=x.device, dtype=x.dtype)
                da = torch.zeros(rows.numel(), self.out_features, device=x.device, dtype=x.dtype)
                zero = torch.zeros(rows.numel(), device=x.device, dtype=torch.long)
                bgmv_shrink(xa, self.A[a:a + 1], ha, zero, scale)
                bgmv_expand(ha, self.B[a:a + 1], da, zero, True)
                delta.index_copy_(0, rows, da)
        return base + delta.reshape(*shp[:-1], self.out_features)


def wrap_decoder_linears(model, K, rank, dtype, device):
    wrapped = 0
    for name, module in list(model.named_modules()):
        if "decoder.layers" not in name:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, child_name,
                        LoRAWrappedLinear(child, K, rank, dtype, device))
                wrapped += 1
    return wrapped


def time_ms(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        s.append(a.elapsed_time(b))
    s.sort()
    return {"median_ms": statistics.median(s), "iqr_ms": s[3 * len(s) // 4] - s[len(s) // 4]}


def check_correctness(model, B, K, device):
    """fp32 reference for the batched delta in the first wrapped linear."""
    lin = next(m for m in model.modules() if isinstance(m, LoRAWrappedLinear))
    x = torch.randn(B, lin.in_features, device=device, dtype=lin.A.dtype).contiguous()
    idx = torch.arange(B, device=device, dtype=torch.long) % K
    delta = torch.zeros(B, lin.out_features, device=device, dtype=lin.A.dtype)
    hbuf = torch.zeros(B, lin.rank, device=device, dtype=lin.A.dtype)
    bgmv_shrink(x, lin.A, hbuf, idx, 1.0)
    bgmv_expand(hbuf, lin.B, delta, idx, True)
    ref = torch.zeros(B, lin.out_features, device=device, dtype=torch.float32)
    xf = x.float()
    for a in range(K):
        rows = (idx == a).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        h = xf.index_select(0, rows) @ lin.A[a].float().t()
        ref.index_add_(0, rows, h @ lin.B[a].float().t())
    return (delta.float() - ref).abs().max().item()


def main():
    p = argparse.ArgumentParser(description="draft-side SGMV in a real draft-model step")
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=1, help="draft tokens per step")
    p.add_argument("--K", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device, dtype = "cuda", torch.float16
    cfg = AutoConfig.from_pretrained(args.model)
    print(f"model=OPT-125m(random-init from config)  hidden={cfg.hidden_size}  "
          f"layers={cfg.num_hidden_layers}  device={torch.cuda.get_device_name(0)}")

    rows_out = []
    for K in args.K:
        torch.manual_seed(0)
        model = OPTForCausalLM(cfg).to(device=device, dtype=dtype).eval()
        n_wrapped = wrap_decoder_linears(model, K, args.rank, dtype, device)
        err = check_correctness(model, args.batch, K, device)

        ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seqlen), device=device)
        STATE["idx"] = (torch.arange(args.batch * args.seqlen, device=device,
                                     dtype=torch.long) % K)

        @torch.inference_mode()
        def fwd(model=model, ids=ids):
            model(ids)

        def run(mode):
            STATE["mode"] = mode
            return time_ms(fwd, args.iters, args.warmup)

        t_none = run("none")
        t_bat = run("batched")
        t_seq = run("seq_bgmv")
        STATE["mode"] = "none"

        base = t_none["median_ms"]
        rec = {
            "K": K, "batch": args.batch, "rank": args.rank,
            "wrapped_linears": n_wrapped,
            "base_ms": base, "batched_ms": t_bat["median_ms"],
            "seq_bgmv_ms": t_seq["median_ms"],
            "batched_overhead_ms": t_bat["median_ms"] - base,
            "batched_overhead_pct": 100 * (t_bat["median_ms"] - base) / base,
            "seq_overhead_ms": t_seq["median_ms"] - base,
            "spd_batched_vs_seq": t_seq["median_ms"] / t_bat["median_ms"],
            "max_abs_err": err,
        }
        rows_out.append(rec)
        print(f"K={K:>3}  base={base:6.3f}  batched={rec['batched_ms']:6.3f} "
              f"(+{rec['batched_overhead_pct']:5.1f}%)  seq_bgmv={rec['seq_bgmv_ms']:7.3f}  "
              f"batched_vs_seq={rec['spd_batched_vs_seq']:5.2f}x  err={err:.1e}")
        del model
        torch.cuda.empty_cache()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"draft_lens_{stamp}.json")
    cpath = os.path.join(outdir, f"draft_lens_{stamp}.csv")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "model": "opt-125m",
                   "results": rows_out}, f, indent=2)
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    max_err = max(r["max_abs_err"] for r in rows_out)
    print(f"\nmax_abs_err: {max_err:.2e}  ({'PASS' if max_err < 5e-2 else 'FAIL'})")
    print(f"written: {jpath}")


if __name__ == "__main__":
    main()
