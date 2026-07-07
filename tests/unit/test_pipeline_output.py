"""Full-size cp.async double-buffer pipeline correctness, across all four
rank variants (8/16/32/64) -- complements test_rank_variants.py's small
D_MODEL/OUT_FEATURES=512 (2 tiles) check with real-model-size dimensions
(D_MODEL=OUT_FEATURES=4096, 16 TILE_K/TILE_O tiles each) and a full
WP_MAX_SEGMENT_TOKENS=32-token segment (8 WP_TOKEN_CHUNK=4 chunks) per
adapter, across K=4 adapters (multi-segment grid.x). A bug confined to a
specific pipeline stage (e.g. wrong double-buffer index past stage 1, or a
race only visible once cp.async actually has multiple tiles in flight) could
pass the 2-tile smoke test and still be wrong here -- this is the test that
would catch it.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

D_MODEL = 4096
OUT_FEATURES = 4096
K = 4
TOKENS_PER_ADAPTER = 32  # = WP_MAX_SEGMENT_TOKENS: full segment, 8 chunks of WP_TOKEN_CHUNK=4


def run_rank(rank, seed):
    torch.manual_seed(seed)
    alpha_scale = 16.0 / rank
    engine = WarpPipeEngine()

    A = torch.randn(K, rank, D_MODEL, device="cuda", dtype=torch.float16) * 0.02
    B = torch.randn(K, OUT_FEATURES, rank, device="cuda", dtype=torch.float16) * 0.02
    for i in range(K):
        engine.register_adapter(i, A[i].contiguous(), B[i].contiguous(), alpha_scale, rank=rank)

    seg_offsets = [i * TOKENS_PER_ADAPTER for i in range(K + 1)]
    engine.build_segments(list(range(K)), seg_offsets)
    total_tokens = K * TOKENS_PER_ADAPTER

    X = torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
    Y_base = torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1

    H = engine.shrink(X, D_MODEL, rank, total_tokens)
    Y = Y_base.clone()
    engine.expand(H, Y, OUT_FEATURES, rank)
    torch.cuda.synchronize()

    H_ref = torch.zeros(total_tokens, rank, device="cuda", dtype=torch.float16)
    Y_ref = Y_base.clone()
    for i in range(K):
        lo, hi = i * TOKENS_PER_ADAPTER, (i + 1) * TOKENS_PER_ADAPTER
        h = alpha_scale * (X[lo:hi].float() @ A[i].float().T)
        H_ref[lo:hi] = h.half()
        y_add = H_ref[lo:hi].float() @ B[i].float().T
        Y_ref[lo:hi] = (Y_ref[lo:hi].float() + y_add).half()

    h_err = (H.float() - H_ref.float()).abs().max().item()
    y_err = (Y.float() - Y_ref.float()).abs().max().item()
    # Per-chunk breakdown: catches a bug confined to one pipeline stage
    # (e.g. only the 2nd half of the tile loop, or only odd chunks) that a
    # single max-over-everything check could still pass by accident if the
    # bug were small relative to other chunks' correct values.
    worst_chunk = None
    worst_chunk_err = -1.0
    for i in range(K):
        for c in range(0, TOKENS_PER_ADAPTER, 4):
            lo, hi = i * TOKENS_PER_ADAPTER + c, i * TOKENS_PER_ADAPTER + min(c + 4, TOKENS_PER_ADAPTER)
            e = (Y[lo:hi].float() - Y_ref[lo:hi].float()).abs().max().item()
            if e > worst_chunk_err:
                worst_chunk_err, worst_chunk = e, (i, c)
    return h_err, y_err, worst_chunk, worst_chunk_err


def _check(rank):
    h_err, y_err, worst_chunk, worst_chunk_err = run_rank(rank, seed=hash(("pipeline", rank)) % (2**31))
    assert h_err < 0.05, f"rank={rank} shrink max abs err {h_err}"
    assert y_err < 0.5, f"rank={rank} expand max abs err {y_err}"
    assert worst_chunk_err < 0.5, f"rank={rank} worst chunk {worst_chunk} err {worst_chunk_err}"
    print(f"PASS rank={rank}: h_err={h_err:.5f} y_err={y_err:.5f} worst_chunk={worst_chunk} ({worst_chunk_err:.5f})")


def test_pipeline_rank_8():
    _check(8)


def test_pipeline_rank_16():
    _check(16)


def test_pipeline_rank_32():
    _check(32)


def test_pipeline_rank_64():
    _check(64)


if __name__ == "__main__":
    test_pipeline_rank_8()
    test_pipeline_rank_16()
    test_pipeline_rank_32()
    test_pipeline_rank_64()
