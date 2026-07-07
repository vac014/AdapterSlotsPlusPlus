import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402
from reference_bgmv import reference_bgmv_expand, reference_bgmv_shrink  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
ALPHA = 16.0
ALPHA_SCALE = ALPHA / RANK


def run_case(K, T):
    torch.manual_seed(K * 1000 + T)
    engine = WarpPipeEngine()
    total_tokens = K * T

    lora_a = torch.randn(K, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    lora_b = torch.randn(K, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    X = torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
    Y_base = torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1

    for i in range(K):
        engine.register_adapter(i, lora_a[i].contiguous(), lora_b[i].contiguous(), ALPHA_SCALE)
    sorted_adapter_ids = list(range(K))
    seg_offsets = [i * T for i in range(K + 1)]
    engine.build_segments(sorted_adapter_ids, seg_offsets)

    H = engine.shrink(X, D_MODEL, RANK, total_tokens)
    Y = Y_base.clone()
    engine.expand(H, Y, OUT_FEATURES, RANK)
    torch.cuda.synchronize()

    # Reference: per-segment, since each segment has its own adapter A/B.
    H_ref = torch.zeros(total_tokens, RANK, dtype=torch.float32, device="cuda")
    Y_ref = Y_base.float().clone()
    for i in range(K):
        t0, t1 = i * T, (i + 1) * T
        H_ref[t0:t1] = reference_bgmv_shrink(X[t0:t1], lora_a[i], ALPHA_SCALE)
        Y_ref[t0:t1] = reference_bgmv_expand(H_ref[t0:t1].half(), lora_b[i], Y_base[t0:t1])

    h_err = (H.float() - H_ref).abs().max().item()
    y_err = (Y.float() - Y_ref).abs().max().item()
    return h_err, y_err


def main():
    all_pass = True
    for K in [1, 2, 4, 8]:
        for T in [1, 4, 8, 16]:
            h_err, y_err = run_case(K, T)
            ok = h_err < 0.05 and y_err < 0.05
            all_pass &= ok
            status = "PASS" if ok else "FAIL"
            print(f"K={K:2d} T={T:2d}  H_err={h_err:.5f}  Y_err={y_err:.5f}  {status}")
    print("ALL PASS" if all_pass else "SOME FAILED")


if __name__ == "__main__":
    main()
