import os
import sys

import torch
import torch.utils.benchmark as bench

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402
from vllm.lora.ops.bgmv_expand import bgmv_expand  # noqa: E402
from vllm.lora.ops.bgmv_shrink import bgmv_shrink  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
ALPHA = 16.0
ALPHA_SCALE = ALPHA / RANK
N_ITERS = 200


def setup(K, T):
    torch.manual_seed(0)
    total_tokens = K * T
    lora_a = torch.randn(K, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    lora_b = torch.randn(K, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    X = torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
    Y_base = torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1
    return lora_a, lora_b, X, Y_base


def bench_warp_pipe(K, T, lora_a, lora_b, X, Y_base):
    engine = WarpPipeEngine()
    total_tokens = K * T
    for i in range(K):
        engine.register_adapter(i, lora_a[i].contiguous(), lora_b[i].contiguous(), ALPHA_SCALE)
    engine.build_segments(list(range(K)), [i * T for i in range(K + 1)])

    def run():
        H = engine.shrink(X, D_MODEL, RANK, total_tokens)
        Y = Y_base.clone()
        engine.expand(H, Y, OUT_FEATURES, RANK)
        return Y

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t = bench.Timer(stmt="run()", globals={"run": run})
    return t.timeit(N_ITERS).mean * 1e6


def bench_vllm_stock(K, T, lora_a, lora_b, X, Y_base):
    total_tokens = K * T
    lora_indices = torch.repeat_interleave(torch.arange(K, device="cuda", dtype=torch.long), T)

    def run():
        H = torch.zeros(total_tokens, RANK, device="cuda", dtype=torch.float16)
        bgmv_shrink(X, lora_a, H, lora_indices, ALPHA_SCALE)
        Y = Y_base.clone()
        bgmv_expand(H, lora_b, Y, lora_indices, add_inputs=True)
        return Y

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t = bench.Timer(stmt="run()", globals={"run": run})
    return t.timeit(N_ITERS).mean * 1e6


def main():
    print(f"{'K':>3} {'T':>3} {'WarpPipe(us)':>14} {'vLLM(us)':>10} {'speedup':>9}")
    speedups = []
    for K in [2, 4, 8, 16]:
        for T in [1, 4, 8, 16]:
            lora_a, lora_b, X, Y_base = setup(K, T)
            wp_us = bench_warp_pipe(K, T, lora_a, lora_b, X, Y_base)
            vllm_us = bench_vllm_stock(K, T, lora_a, lora_b, X, Y_base)
            speedup = vllm_us / wp_us
            speedups.append(speedup)
            print(f"{K:>3} {T:>3} {wp_us:>14.2f} {vllm_us:>10.2f} {speedup:>8.3f}x")
    print(f"\nmean speedup: {sum(speedups)/len(speedups):.3f}x  "
          f"min: {min(speedups):.3f}x  max: {max(speedups):.3f}x")


if __name__ == "__main__":
    main()
