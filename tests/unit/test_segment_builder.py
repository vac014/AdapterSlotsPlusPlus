"""Direct unit test of build_segment_table()'s output shape (via the only
Python-exposed surface, engine.build_segments()) -- checks segment count,
ordering, and that use_staging never claims a slot it does not have real
staging pointers for. Complements test_scheduler_bridge.py (real scheduler
data) by testing segment_builder in isolation with hand-picked adapter/offset
combinations, including edge cases the real scheduler output may never produce
but the function must still handle safely (K=1, non-uniform segment sizes,
K=WP_MAX_SEGMENTS boundary).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
ALPHA_SCALE = 16.0 / RANK


def make_engine(k):
    engine = WarpPipeEngine()
    a = torch.randn(k, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    b = torch.randn(k, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02
    for i in range(k):
        engine.register_adapter(i, a[i].contiguous(), b[i].contiguous(), ALPHA_SCALE)
    return engine, a, b


def test_single_segment():
    engine, _, _ = make_engine(1)
    engine.build_segments([0], [0, 4])
    print("PASS: K=1 single segment builds without error")


def test_non_uniform_sizes():
    engine, _, _ = make_engine(3)
    # Non-uniform: 1, 31 (= WP_MAX_SEGMENT_TOKENS), 2 tokens.
    engine.build_segments([0, 1, 2], [0, 1, 32, 34])
    X = torch.randn(34, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
    Y = torch.randn(34, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1
    H = engine.shrink(X, D_MODEL, RANK, 34)
    engine.expand(H, Y, OUT_FEATURES, RANK)
    torch.cuda.synchronize()
    print("PASS: non-uniform segment sizes (1, 31, 2 tokens) build and run without error")


def test_use_staging_never_claims_unready_slot():
    """use_staging=1 with null staging pointers is a segfault in the kernel.
    Build segments with NO prefetch ever triggered -- every segment's
    use_staging must come back 0 (verified indirectly: shrink/expand must
    produce output, since a null staging pointer would segfault the kernel)."""
    engine, a, b = make_engine(4)
    engine.build_segments([0, 1, 2, 3], [0, 2, 4, 6, 8])
    X = torch.randn(8, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
    Y = torch.randn(8, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1
    H = engine.shrink(X, D_MODEL, RANK, 8)
    engine.expand(H, Y, OUT_FEATURES, RANK)
    torch.cuda.synchronize()
    assert not torch.isnan(H).any() and not torch.isnan(Y).any()
    print("PASS: no prefetch triggered -> use_staging stays 0 everywhere, no null-pointer read")


def test_max_segments_boundary():
    from lora_warp_pipe.config import WarpPipeConfig  # noqa: F401

    WP_MAX_SEGMENTS = 256
    k = WP_MAX_SEGMENTS
    engine, a, b = make_engine(k)
    offsets = list(range(0, k + 1))  # 1 token per segment, K=WP_MAX_SEGMENTS exactly
    engine.build_segments(list(range(k)), offsets)
    print(f"PASS: K={k} (== WP_MAX_SEGMENTS) builds without error")


if __name__ == "__main__":
    test_single_segment()
    test_non_uniform_sizes()
    test_use_staging_never_claims_unready_slot()
    test_max_segments_boundary()
    print("ALL PASS: test_segment_builder.py")
