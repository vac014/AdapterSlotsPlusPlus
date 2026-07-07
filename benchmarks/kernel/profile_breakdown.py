import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
ALPHA = 16.0
ALPHA_SCALE = ALPHA / RANK
N_ITERS = 50
N_WARMUP = 10


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    torch.manual_seed(0)
    total_tokens = K * T
    engine = WarpPipeEngine()
    lora_a = torch.randn(K, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    lora_b = torch.randn(K, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    X = torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
    Y_base = torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1
    for i in range(K):
        engine.register_adapter(i, lora_a[i].contiguous(), lora_b[i].contiguous(), ALPHA_SCALE)
    engine.build_segments(list(range(K)), [i * T for i in range(K + 1)])

    shrink_ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITERS)]
    shrink_ev_end = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITERS)]
    expand_ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITERS)]
    expand_ev_end = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITERS)]

    for _ in range(N_WARMUP):
        H = engine.shrink(X, D_MODEL, RANK, total_tokens)
        Y = Y_base.clone()
        engine.expand(H, Y, OUT_FEATURES, RANK)
    torch.cuda.synchronize()

    for i in range(N_ITERS):
        shrink_ev_start[i].record()
        H = engine.shrink(X, D_MODEL, RANK, total_tokens)
        shrink_ev_end[i].record()
        Y = Y_base.clone()
        expand_ev_start[i].record()
        engine.expand(H, Y, OUT_FEATURES, RANK)
        expand_ev_end[i].record()
    torch.cuda.synchronize()

    shrink_us = sum(s.elapsed_time(e) for s, e in zip(shrink_ev_start, shrink_ev_end)) / N_ITERS * 1000
    expand_us = sum(s.elapsed_time(e) for s, e in zip(expand_ev_start, expand_ev_end)) / N_ITERS * 1000
    print(f"K={K} T={T}  shrink={shrink_us:.2f}us  expand={expand_us:.2f}us  ratio(expand/shrink)={expand_us/shrink_us:.3f}x")


if __name__ == "__main__":
    main()
