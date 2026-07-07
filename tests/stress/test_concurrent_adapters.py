"""K=32 concurrent adapters, randomized per-adapter token counts, many iterations.

Correctness is checked against an fp32 reference on every iteration rather than only
checking that nothing crashed. The bugs this kernel is actually prone to (wrong
scaling, off-by-one segment boundaries) produce clean runs with wrong numbers under
unusual but valid shapes, and a crash-only stress test would pass straight through
them.
"""
import random
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "correctness"))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402
from reference_bgmv import reference_bgmv_expand, reference_bgmv_shrink  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
ALPHA_SCALE = 16.0 / RANK
K = 32
N_ITERS = 25


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    engine = WarpPipeEngine()

    lora_a = torch.randn(K, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    lora_b = torch.randn(K, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    for i in range(K):
        engine.register_adapter(i, lora_a[i].contiguous(), lora_b[i].contiguous(), ALPHA_SCALE)

    max_err_h, max_err_y = 0.0, 0.0
    for it in range(N_ITERS):
        # Randomized arrival: a random subset of the 32 adapters is active
        # this "step", each with a random token count (1-32, since
        # WP_MAX_SEGMENT_TOKENS=32 is this kernel's per-segment cap).
        active = rng.sample(range(K), rng.randint(1, K))
        tokens_per = {aid: rng.randint(1, 32) for aid in active}
        seg_offsets = [0]
        for aid in active:
            seg_offsets.append(seg_offsets[-1] + tokens_per[aid])
        total_tokens = seg_offsets[-1]

        engine.build_segments(active, seg_offsets)
        X = torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
        Y_base = torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1
        H = engine.shrink(X, D_MODEL, RANK, total_tokens)
        Y = Y_base.clone()
        engine.expand(H, Y, OUT_FEATURES, RANK)
        torch.cuda.synchronize()

        H_ref = torch.zeros(total_tokens, RANK, dtype=torch.float32, device="cuda")
        Y_ref = Y_base.float().clone()
        for i, aid in enumerate(active):
            t0, t1 = seg_offsets[i], seg_offsets[i + 1]
            H_ref[t0:t1] = reference_bgmv_shrink(X[t0:t1], lora_a[aid], ALPHA_SCALE)
            Y_ref[t0:t1] = reference_bgmv_expand(H_ref[t0:t1].half(), lora_b[aid], Y_base[t0:t1])

        h_err = (H.float() - H_ref).abs().max().item()
        y_err = (Y.float() - Y_ref).abs().max().item()
        max_err_h, max_err_y = max(max_err_h, h_err), max(max_err_y, y_err)
        assert h_err < 0.05 and y_err < 0.05, (
            f"iter {it}: incorrect output, active={active}, tokens_per={tokens_per}, h_err={h_err}, y_err={y_err}")

    print(f"PASS: {N_ITERS} iterations, K=32 pool, random subsets + random token counts each iter, "
          f"max_h_err={max_err_h:.5f} max_y_err={max_err_y:.5f}, no crash, no incorrect output")


if __name__ == "__main__":
    main()
