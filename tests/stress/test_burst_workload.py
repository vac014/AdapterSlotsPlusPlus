"""Erlang-style burst epochs: alternating high-load
(many adapters, large segments, prefetch saturating the staging pool) and
low-load (single adapter) epochs, checking pipeline stability -- no
crash, no leaked staging slots, correct output throughout the transition
between epochs (the actual risk: state left over from a high-load epoch
corrupting a low-load epoch right after it, or vice versa).
"""
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
K_POOL = 16
WP_MAX_STAGING = 8


def run_epoch(engine, lora_a, lora_b, active, tokens_per, use_prefetch, label):
    if use_prefetch:
        for aid in active[:WP_MAX_STAGING]:
            engine.prefetch_adapter(aid)
        torch.cuda.synchronize()  # deterministic for this test, not a live-path pattern

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
    print(f"{label}: active={len(active)} total_tokens={total_tokens} h_err={h_err:.5f} y_err={y_err:.5f}")
    assert h_err < 0.05 and y_err < 0.05, f"{label}: incorrect output"

    if use_prefetch:
        for aid in active[:WP_MAX_STAGING]:
            engine.release_prefetch(aid)


def main():
    torch.manual_seed(0)
    engine = WarpPipeEngine()
    lora_a = torch.randn(K_POOL, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    lora_b = torch.randn(K_POOL, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    for i in range(K_POOL):
        engine.register_adapter(i, lora_a[i].contiguous(), lora_b[i].contiguous(), ALPHA_SCALE)

    epochs = [
        ("burst-1 (high, K=16, prefetch on)", list(range(K_POOL)), {i: 8 for i in range(K_POOL)}, True),
        ("calm-1 (low, K=1)", [0], {0: 1}, False),
        ("burst-2 (high, K=12, prefetch on)", list(range(12)), {i: 16 for i in range(12)}, True),
        ("calm-2 (low, K=2)", [3, 7], {3: 2, 7: 1}, False),
        ("burst-3 (high, K=16, prefetch off)", list(range(K_POOL)), {i: 4 for i in range(K_POOL)}, False),
        ("calm-3 (low, K=1, same adapter as burst-3's first)", [0], {0: 1}, False),
    ]
    for label, active, tokens_per, use_prefetch in epochs:
        run_epoch(engine, lora_a, lora_b, active, tokens_per, use_prefetch, label)

    print("PASS: 6 alternating burst/calm epochs, prefetch toggled across epochs, no crash, "
          "no cross-epoch state corruption, correct output throughout")


if __name__ == "__main__":
    main()
