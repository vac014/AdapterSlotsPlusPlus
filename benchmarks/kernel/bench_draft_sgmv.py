#!/usr/bin/env python3
"""
Draft-side SGMV in isolation: is one batched pass flat in K?

Mirrors the draft path at decode time: B rows, one draft token each, every row routed to one
of K adapters. Three implementations of the same delta y += scale * (x A^T) B^T:

  batched   one bgmv_shrink and one bgmv_expand over the whole batch with a per-row adapter
            index. One kernel pair regardless of K. This is draft-side SGMV.
  seq_bgmv  the same kernel called once per adapter over that adapter's rows, so K launches.
            Isolates launch and serialization overhead.
  seq_torch per-adapter torch matmuls.

Correctness is checked against an fp32 reference. A llama-7b-dimension GEMM stands in for the
target decode step so the draft-apply cost can be read as a fraction of one target step.

Weights are random: this is a cost and correctness claim, not a quality one.
"""

import argparse
import csv
import json
import os
import statistics
import time
from typing import Dict, List

import torch

from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.bgmv_expand import bgmv_expand


def _sync():
    torch.cuda.synchronize()


def time_ms(fn, iters: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))  # ms
    samples.sort()
    med = statistics.median(samples)
    q1 = samples[len(samples) // 4]
    q3 = samples[(3 * len(samples)) // 4]
    return {"median_ms": med, "iqr_ms": q3 - q1, "min_ms": samples[0]}


def make_case(B, K, d_in, d_out, rank, device, dtype, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, d_in, device=device, dtype=dtype, generator=g).contiguous()
    # A: [K, rank, d_in]   B: [K, d_out, rank]  (vLLM bgmv layout)
    A = (torch.randn(K, rank, d_in, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    Bw = (torch.randn(K, d_out, rank, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
    # round-robin row -> adapter
    idx = torch.arange(B, device=device, dtype=torch.long) % K
    return x, A, Bw, idx


def run_batched(x, A, Bw, idx, scale, y, hbuf):
    hbuf.zero_()
    bgmv_shrink(x, A, hbuf, idx, scale)
    bgmv_expand(hbuf, Bw, y, idx, True)


def run_seq_bgmv(x, A, Bw, idx, scale, y, hbuf):
    K = A.size(0)
    hbuf.zero_()
    for a in range(K):
        rows = (idx == a).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        xa = x.index_select(0, rows).contiguous()
        ha = hbuf.index_select(0, rows).contiguous()
        ya = y.index_select(0, rows).contiguous()
        zero_idx = torch.zeros(rows.numel(), device=x.device, dtype=torch.long)
        bgmv_shrink(xa, A[a:a + 1], ha, zero_idx, scale)
        bgmv_expand(ha, Bw[a:a + 1], ya, zero_idx, True)
        y.index_copy_(0, rows, ya)


def run_seq_torch(x, A, Bw, idx, scale, y):
    K = A.size(0)
    for a in range(K):
        rows = (idx == a).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        xa = x.index_select(0, rows)
        ha = xa @ A[a].t()           # [rows, rank]
        ya = (ha @ Bw[a].t()) * scale  # [rows, d_out]
        y.index_add_(0, rows, ya)


def reference(x, A, Bw, idx, scale, d_out):
    B = x.size(0)
    y = torch.zeros(B, d_out, device=x.device, dtype=torch.float32)
    xf = x.float()
    for a in range(A.size(0)):
        rows = (idx == a).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        ha = xf.index_select(0, rows) @ A[a].float().t()
        ya = (ha @ Bw[a].float().t()) * scale
        y.index_add_(0, rows, ya)
    return y


def target_step_proxy_ms(B, device, dtype, iters, warmup,
                         hidden=4096, inter=11008, layers=32):
    """One decode-step base-GEMM proxy for llama-7b: the dominant per-layer
    linear projections (q,k,v,o,gate,up,down) for B rows, x layers. Ignores
    attention/norm; base GEMMs dominate decode compute. Fair upper-bound-ish
    denominator for the draft/target ratio."""
    g = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(B, hidden, device=device, dtype=dtype, generator=g)
    Wq = torch.randn(hidden, hidden, device=device, dtype=dtype, generator=g)
    Wk = torch.randn(hidden, hidden, device=device, dtype=dtype, generator=g)
    Wv = torch.randn(hidden, hidden, device=device, dtype=dtype, generator=g)
    Wo = torch.randn(hidden, hidden, device=device, dtype=dtype, generator=g)
    Wg = torch.randn(hidden, inter, device=device, dtype=dtype, generator=g)
    Wu = torch.randn(hidden, inter, device=device, dtype=dtype, generator=g)
    Wd = torch.randn(inter, hidden, device=device, dtype=dtype, generator=g)

    def one_step():
        h = x
        for _ in range(layers):
            _ = h @ Wq; _ = h @ Wk; _ = h @ Wv
            _ = h @ Wo
            gg = h @ Wg; uu = h @ Wu
            _ = (gg * uu) @ Wd
    return time_ms(one_step, iters, warmup)


def main():
    p = argparse.ArgumentParser(description="draft-side SGMV microbenchmark")
    p.add_argument("--batch", type=int, default=64, help="B: concurrent draft rows")
    p.add_argument("--K", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--rank", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--d", type=int, nargs="+", default=[1024, 2048], help="draft hidden")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--tol", type=float, default=5e-2, help="max_abs_err gate")
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    device = "cuda"
    dtype = getattr(torch, args.dtype)
    dev_name = torch.cuda.get_device_name(0)
    print(f"device={dev_name}  dtype={args.dtype}  B={args.batch}  "
          f"iters={args.iters}  warmup={args.warmup}")

    # target-step proxy (independent of K/rank/d): one number for the ratio
    tgt = target_step_proxy_ms(args.batch, device, dtype, args.iters, args.warmup)
    print(f"\ntarget decode-step proxy (llama-7b base GEMMs, B={args.batch}): "
          f"{tgt['median_ms']:.3f} ms\n")

    rows_out: List[Dict] = []
    header = ("d", "rank", "K", "batched_ms", "seq_bgmv_ms", "seq_torch_ms",
              "spd_vs_seqbgmv", "spd_vs_seqtorch", "draft/target", "max_abs_err")
    print("  " + "  ".join(f"{h:>13}" for h in header))

    for d in args.d:
        d_out = d
        for rank in args.rank:
            for K in args.K:
                if K > args.batch:
                    continue
                x, A, Bw, idx = make_case(args.batch, K, d, d_out, rank, device, dtype)
                y_ref = reference(x, A, Bw, idx, args.scale, d_out)

                hbuf = torch.zeros(args.batch, rank, device=device, dtype=dtype)
                y_b = torch.zeros(args.batch, d_out, device=device, dtype=dtype)
                run_batched(x, A, Bw, idx, args.scale, y_b, hbuf)
                err = (y_b.float() - y_ref).abs().max().item()

                y0 = torch.zeros(args.batch, d_out, device=device, dtype=dtype)
                t_b = time_ms(lambda: run_batched(x, A, Bw, idx, args.scale,
                              y0.zero_(), hbuf), args.iters, args.warmup)
                t_sb = time_ms(lambda: run_seq_bgmv(x, A, Bw, idx, args.scale,
                               y0.zero_(), hbuf), args.iters, args.warmup)
                t_st = time_ms(lambda: run_seq_torch(x, A, Bw, idx, args.scale,
                               y0.zero_()), args.iters, args.warmup)

                spd_sb = t_sb["median_ms"] / t_b["median_ms"]
                spd_st = t_st["median_ms"] / t_b["median_ms"]
                ratio = t_b["median_ms"] / tgt["median_ms"]
                rec = {
                    "d": d, "rank": rank, "K": K, "batch": args.batch,
                    "batched_ms": t_b["median_ms"], "batched_iqr": t_b["iqr_ms"],
                    "seq_bgmv_ms": t_sb["median_ms"], "seq_torch_ms": t_st["median_ms"],
                    "spd_vs_seqbgmv": spd_sb, "spd_vs_seqtorch": spd_st,
                    "draft_target_ratio": ratio, "max_abs_err": err,
                    "target_step_ms": tgt["median_ms"],
                }
                rows_out.append(rec)
                vals = (d, rank, K, f"{t_b['median_ms']:.4f}",
                        f"{t_sb['median_ms']:.4f}", f"{t_st['median_ms']:.4f}",
                        f"{spd_sb:.2f}x", f"{spd_st:.2f}x", f"{ratio:.4f}",
                        f"{err:.1e}")
                print("  " + "  ".join(f"{str(v):>13}" for v in vals))

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"draft_sgmv_{stamp}.json")
    cpath = os.path.join(outdir, f"draft_sgmv_{stamp}.csv")
    with open(jpath, "w") as f:
        json.dump({"device": dev_name, "dtype": args.dtype, "batch": args.batch,
                   "target_step_ms": tgt["median_ms"], "results": rows_out}, f, indent=2)
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    max_err = max(r["max_abs_err"] for r in rows_out)
    print(f"\nmax_abs_err across grid: {max_err:.2e}  (gate {args.tol:.0e}: "
          f"{'PASS' if max_err <= args.tol else 'FAIL'})")
    print(f"written: {jpath}\n         {cpath}")


if __name__ == "__main__":
    main()
