#!/usr/bin/env python3
"""
CUDA-graph capture of a LoRA draft step, and cross-adapter batched verify.

Eager, a batched draft-LoRA step is flat in K but pays a large per-op launch cost. Capturing it
is what removes that, and a captured draft step with LoRA baked in and per-step-varying adapter
indices is the capability a stock speculative stack does not have.

Part A captures a multi-layer linear backbone with a per-request LoRA delta on every linear
into a CUDA graph over static buffers, and replays it with fresh inputs and fresh adapter
indices. Two things are validated: that a replay with different adapter indices than capture
matches an eager recompute for the replay-time indices, so nothing from capture is baked in;
and that replay collapses the eager per-op overhead.

Part B verifies B requests, each proposing gamma+1 tokens, in one batched pass with each row on
its own adapter, checked against eager and exercising ragged per-request truncation at the first
reject.

Random adapters: this is a correctness and performance claim, not an acceptance one.
"""

import argparse
import json
import os
import statistics
import time

import torch

from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.bgmv_expand import bgmv_expand


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
    return statistics.median(s)


class GraphDraftStep:
    """Multi-layer LoRA-augmented linear backbone with static buffers, so the
    whole forward can be CUDA-graph captured and replayed with new data."""

    def __init__(self, N, d, layers, K, rank, dtype, device, scale=1.0, seed=0,
                 res_scale=0.3):
        g = torch.Generator(device=device).manual_seed(seed)
        self.N, self.d, self.L, self.K, self.rank, self.scale = N, d, layers, K, rank, scale
        self.res_scale = res_scale  # sub-unit residual gain keeps magnitudes bounded
        mk = lambda *s: (torch.randn(*s, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
        self.W1 = [mk(d, d) for _ in range(layers)]
        self.W2 = [mk(d, d) for _ in range(layers)]
        self.A1 = [mk(K, rank, d) for _ in range(layers)]
        self.B1 = [mk(K, d, rank) for _ in range(layers)]
        self.A2 = [mk(K, rank, d) for _ in range(layers)]
        self.B2 = [mk(K, d, rank) for _ in range(layers)]
        # static buffers
        self.x0 = torch.zeros(N, d, device=device, dtype=dtype)   # input (set per replay)
        self.idx = torch.zeros(N, device=device, dtype=torch.long)  # adapter index (set per replay)
        self.xb = torch.zeros(N, d, device=device, dtype=dtype)   # working
        self.sc = torch.zeros(N, d, device=device, dtype=dtype)   # scratch
        self.hb = torch.zeros(N, rank, device=device, dtype=dtype)  # shrink buffer
        self.out = torch.zeros(N, d, device=device, dtype=dtype)
        self.graph = None

    def _sublayer(self, W, A, B):
        torch.mm(self.xb, W.t(), out=self.sc)      # base
        self.hb.zero_()
        bgmv_shrink(self.xb, A, self.hb, self.idx, self.scale)
        bgmv_expand(self.hb, B, self.sc, self.idx, True)   # sc = base + lora
        self.xb.add_(self.sc, alpha=self.res_scale)        # residual (bounded gain)

    def _forward_inplace(self):
        self.xb.copy_(self.x0)
        for l in range(self.L):
            self._sublayer(self.W1[l], self.A1[l], self.B1[l])
            self._sublayer(self.W2[l], self.A2[l], self.B2[l])
        self.out.copy_(self.xb)

    def eager(self, x, idx):
        self.x0.copy_(x); self.idx.copy_(idx)
        self._forward_inplace()
        return self.out.clone()

    def capture(self, x, idx):
        self.x0.copy_(x); self.idx.copy_(idx)
        for _ in range(3):              # warmup: JIT/config-cache before capture
            self._forward_inplace()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._forward_inplace()

    def replay(self, x, idx):
        self.x0.copy_(x); self.idx.copy_(idx)
        self.graph.replay()
        return self.out


def part_a(args, device, dtype):
    print("\n=== Part A: CUDA-graph-captured LoRA draft step ===")
    rows = []
    for K in args.K:
        m = GraphDraftStep(args.batch, args.d, args.layers, K, args.rank, dtype, device)
        g = torch.Generator(device=device).manual_seed(1)
        x_cap = torch.randn(args.batch, args.d, device=device, dtype=dtype, generator=g)
        idx_cap = torch.arange(args.batch, device=device) % K
        # different data + different adapter assignment for replay
        x_rep = torch.randn(args.batch, args.d, device=device, dtype=dtype, generator=g)
        idx_rep = torch.randperm(args.batch, device=device, generator=g) % K

        m.capture(x_cap, idx_cap)
        out_replay = m.replay(x_rep, idx_rep).clone()
        out_eager = m.eager(x_rep, idx_rep)          # recompute with replay-time data
        abs_err = (out_replay.float() - out_eager.float()).abs().max().item()
        rel_err = abs_err / (out_eager.float().abs().max().item() + 1e-6)

        t_eager = time_ms(lambda: m.eager(x_rep, idx_rep), args.iters, args.warmup)
        t_graph = time_ms(lambda: m.replay(x_rep, idx_rep), args.iters, args.warmup)
        rec = {"part": "A", "K": K, "batch": args.batch, "d": args.d,
               "layers": args.layers, "rank": args.rank,
               "eager_ms": t_eager, "graph_ms": t_graph,
               "graph_speedup": t_eager / t_graph,
               "replay_vs_eager_abs_err": abs_err, "replay_vs_eager_rel_err": rel_err}
        rows.append(rec)
        print(f"K={K:>3}  eager={t_eager:7.3f}ms  graph={t_graph:7.3f}ms  "
              f"speedup={rec['graph_speedup']:5.2f}x  "
              f"graph-safety rel_err(diff idx)={rel_err:.1e}  "
              f"{'SAFE' if rel_err < 1e-2 else 'UNSAFE'}")
    return rows


def cross_adapter_verify(x, A, B, row_idx, scale):
    """Batched verify: apply each row's adapter LoRA in one BGMV pass."""
    n, d = x.shape
    r = A.size(1)
    h = torch.zeros(n, r, device=x.device, dtype=x.dtype)
    y = torch.zeros(n, d, device=x.device, dtype=x.dtype)
    bgmv_shrink(x, A, h, row_idx, scale)
    bgmv_expand(h, B, y, row_idx, True)
    return y


def part_b(args, device, dtype):
    print("\n=== Part B: cross-adapter batched verify (gamma+1 rows/request) ===")
    rows = []
    gamma = args.gamma
    for K in args.K:
        B_req = args.batch
        n = B_req * (gamma + 1)
        g = torch.Generator(device=device).manual_seed(2)
        x = torch.randn(n, args.d, device=device, dtype=dtype, generator=g).contiguous()
        A = (torch.randn(K, args.rank, args.d, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
        Bw = (torch.randn(K, args.d, args.rank, device=device, dtype=dtype, generator=g) * 0.02).contiguous()
        # each request -> one adapter; all its gamma+1 rows share it
        req_adapter = torch.arange(B_req, device=device) % K
        row_idx = req_adapter.repeat_interleave(gamma + 1)

        y = cross_adapter_verify(x, A, Bw, row_idx, 1.0)
        # fp32 reference
        ref = torch.zeros(n, args.d, device=device, dtype=torch.float32)
        xf = x.float()
        for a in range(K):
            rmask = (row_idx == a).nonzero(as_tuple=True)[0]
            if rmask.numel() == 0:
                continue
            h = xf.index_select(0, rmask) @ A[a].float().t()
            ref.index_add_(0, rmask, h @ Bw[a].float().t())
        err = (y.float() - ref).abs().max().item()

        # ragged acceptance: synthetic accept mask, count accepted per request
        acc = (torch.rand(B_req, gamma + 1, device=device, generator=g) > 0.3)
        first_rej = torch.where(acc.all(1), torch.full((B_req,), gamma + 1, device=device),
                                (~acc).float().argmax(1))
        accepted = int(first_rej.clamp(max=gamma).sum().item())

        t = time_ms(lambda: cross_adapter_verify(x, A, Bw, row_idx, 1.0), args.iters, args.warmup)
        rec = {"part": "B", "K": K, "requests": B_req, "gamma": gamma, "rows": n,
               "verify_ms": t, "max_abs_err": err, "accepted_tokens": accepted}
        rows.append(rec)
        print(f"K={K:>3}  rows={n:>4}  verify={t:6.3f}ms  err={err:.1e}  "
              f"accepted(synthetic)={accepted}/{B_req*gamma}")
    return rows


def main():
    p = argparse.ArgumentParser(description="graph-captured LoRA draft + cross-adapter verify")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d", type=int, default=2048, help="draft hidden width")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--K", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                   "..", "..", "results"))
    args = p.parse_args()

    assert torch.cuda.is_available()
    device, dtype = "cuda", torch.float16
    print(f"device={torch.cuda.get_device_name(0)}  d={args.d}  layers={args.layers}  "
          f"batch={args.batch}  rank={args.rank}  gamma={args.gamma}")

    rows = part_a(args, device, dtype) + part_b(args, device, dtype)

    outdir = os.path.abspath(args.outdir); os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(outdir, f"graph_capture_{stamp}.json")
    with open(jpath, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "results": rows}, f, indent=2)
    a_rows = [r for r in rows if r["part"] == "A"]
    max_unsafe = max(r["replay_vs_eager_rel_err"] for r in a_rows)
    print(f"\ngraph-safety max rel_err: {max_unsafe:.2e}  "
          f"({'ALL SAFE' if max_unsafe < 1e-2 else 'UNSAFE'})")
    print(f"written: {jpath}")


if __name__ == "__main__":
    main()
