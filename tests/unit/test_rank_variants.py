"""Correctness sweep for the r8/r16/r32/r64 cp.async pipeline shrink/expand
kernels (Task #109) -- warp_pipe_rgen.cuh generalizes warp_pipe_r32.cu's
hand-specialized RANK==warpSize==32 kernels to other ranks via template<int
RANK>, with each lane owning ceil(RANK/32) rank-columns instead of exactly
one. This test exists because that generalization is new code with no prior
validation -- the r32 path was already cross-checked against vLLM's real
bgmv_shrink/bgmv_expand elsewhere in this suite, but r8/r16/r64 have not been
checked against anything until now.

Reference: a plain PyTorch einsum computing the same
  H = alpha_scale * X @ A^T            (shrink)
  Y = Y_base + H @ B^T                 (expand, B stored [out_features, rank])
matching the convention documented in warp_pipe_r32.cu's bgmv_expand_kernel
comment (alpha_scale applied exactly once, in shrink only).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

D_MODEL = 512
OUT_FEATURES = 512
K = 3
TOKENS_PER_ADAPTER = 17  # deliberately not a multiple of WP_TOKEN_CHUNK(4) or 32, to exercise tail handling


def run_rank(rank):
    torch.manual_seed(0)
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

    # Reference, computed per-adapter in fp32 then cast, matching the
    # kernel's own fp32-accumulate-then-cast-to-fp16 pattern.
    H_ref = torch.zeros(total_tokens, rank, device="cuda", dtype=torch.float16)
    Y_ref = Y_base.clone()
    for i in range(K):
        lo, hi = i * TOKENS_PER_ADAPTER, (i + 1) * TOKENS_PER_ADAPTER
        x_slice = X[lo:hi].float()
        h = alpha_scale * (x_slice @ A[i].float().T)
        H_ref[lo:hi] = h.half()
        y_add = H_ref[lo:hi].float() @ B[i].float().T
        Y_ref[lo:hi] = (Y_ref[lo:hi].float() + y_add).half()

    h_err = (H.float() - H_ref.float()).abs().max().item()
    y_err = (Y.float() - Y_ref.float()).abs().max().item()
    return h_err, y_err


def test_rank_8():
    h_err, y_err = run_rank(8)
    assert h_err < 0.05, f"rank=8 shrink max abs err {h_err}"
    assert y_err < 0.5, f"rank=8 expand max abs err {y_err}"
    print(f"PASS rank=8: h_err={h_err:.5f} y_err={y_err:.5f}")


def test_rank_16():
    h_err, y_err = run_rank(16)
    assert h_err < 0.05, f"rank=16 shrink max abs err {h_err}"
    assert y_err < 0.5, f"rank=16 expand max abs err {y_err}"
    print(f"PASS rank=16: h_err={h_err:.5f} y_err={y_err:.5f}")


def test_rank_32():
    h_err, y_err = run_rank(32)
    assert h_err < 0.05, f"rank=32 shrink max abs err {h_err}"
    assert y_err < 0.5, f"rank=32 expand max abs err {y_err}"
    print(f"PASS rank=32: h_err={h_err:.5f} y_err={y_err:.5f}")


def test_rank_64():
    h_err, y_err = run_rank(64)
    assert h_err < 0.05, f"rank=64 shrink max abs err {h_err}"
    assert y_err < 0.5, f"rank=64 expand max abs err {y_err}"
    print(f"PASS rank=64: h_err={h_err:.5f} y_err={y_err:.5f}")


def test_rank_supported_table():
    engine = WarpPipeEngine()
    for r in (8, 16, 32, 64):
        assert engine._ext.rank_supported(r), f"rank_supported({r}) should be True"
    for r in (4, 24, 48, 128):
        assert not engine._ext.rank_supported(r), f"rank_supported({r}) should be False"
    print("PASS: rank_supported() table matches {8,16,32,64}")


if __name__ == "__main__":
    test_rank_8()
    test_rank_16()
    test_rank_32()
    test_rank_64()
    test_rank_supported_table()
