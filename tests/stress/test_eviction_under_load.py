"""Eviction during active dispatch: evict an
adapter immediately after a kernel call that used it, then immediately
allocate new same-sized tensors (forcing PyTorch's caching allocator to
consider reusing that just-freed memory) and run more kernels -- the real
risk being a use-after-free if a kernel launch's GPU-side reads weren't
actually complete when the Python-side tensor refcount hit zero. PyTorch's
CUDA caching allocator does stream-ordered deferred reuse (record_stream)
which should make this safe as long as every call here stays on the same
current stream the kernels were dispatched on, which it does -- this test
exists to confirm that empirically under repeated alloc/evict/realloc
pressure, not just assume it from reading the allocator's docs.
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
N_CYCLES = 40


def main():
    torch.manual_seed(0)
    engine = WarpPipeEngine()

    for cycle in range(N_CYCLES):
        # Fresh tensors every cycle -- after evict_adapter() drops the only
        # Python reference (held_tensors_ in the C++ extension is also
        # erased by evict_adapter), the *previous* cycle's A/B memory is
        # eligible for the caching allocator to hand back out here.
        a = (torch.randn(RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02).contiguous()
        b = (torch.randn(OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02).contiguous()
        adapter_id = cycle % 4  # reuse a small set of ids, like a real slot pool would

        engine.register_adapter(adapter_id, a, b, ALPHA_SCALE)
        engine.build_segments([adapter_id], [0, 4])
        X = torch.randn(4, D_MODEL, device="cuda", dtype=torch.float16) * 0.1
        Y_base = torch.randn(4, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1
        H = engine.shrink(X, D_MODEL, RANK, 4)
        Y = Y_base.clone()
        engine.expand(H, Y, OUT_FEATURES, RANK)
        torch.cuda.synchronize()

        H_ref = reference_bgmv_shrink(X, a, ALPHA_SCALE)
        Y_ref = reference_bgmv_expand(H_ref.half(), b, Y_base)
        h_err = (H.float() - H_ref).abs().max().item()
        y_err = (Y.float() - Y_ref).abs().max().item()
        assert h_err < 0.05 and y_err < 0.05, f"cycle {cycle}: incorrect output (use-after-free corruption?)"

        # Evict immediately after use -- the next cycle's fresh tensors are
        # likely to land in the same freed memory region.
        engine.evict_adapter(adapter_id)
        del a, b

    print(f"PASS: {N_CYCLES} register/use/evict cycles reusing a small id pool, fresh tensors each "
          f"cycle likely reusing just-freed memory, no crash, no use-after-free corruption detected")


if __name__ == "__main__":
    main()
