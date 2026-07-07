"""Standalone correctness test for build_seg_bounds()+add_lora_packed()
against vLLM's own PunicaWrapper.add_lora() semantics (y[i] += x[i] @
A[idx[i]].T @ B[idx[i]].T * scale, no-op where idx[i] < 0), before wiring it
into the live FusedPunicaWrapperWarpPipe class. Cheaper/faster to iterate on
than a full model load, and isolates kernel-interface bugs from
model-integration bugs.

The old single-call API is split in two:
build_seg_bounds(token_lora_indices) builds segment boundaries once;
add_lora_packed(y, x, wa_t_all, wb_t_all, scale) (no indices arg anymore)
resolves THIS call's A_ptr/B_ptr and runs DSAK shrink/expand against
whatever boundaries are currently in meta_dev_. Every run_case() call below
does both steps, mirroring update_metadata()-then-add_lora() in the live
path. test_segment_reuse_across_layers below additionally checks the
property the once-per-step boundary build depends on: boundaries built once stay correct across multiple
add_lora_packed() calls with DIFFERENT wa_t_all/wb_t_all (different "layers"),
the way one real decode step's ~80 layer calls all reuse one boundary build.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lora_warp_pipe import WarpPipeEngine  # noqa: E402

RANK = 32
D_MODEL = 4096
OUT_FEATURES = 4096
NUM_LORAS = 8
ALPHA_SCALE = 16.0 / RANK


def reference_add_lora(y, x, wa_t_all, wb_t_all, scale, indices):
    y_ref = y.float().clone()
    for i in range(x.shape[0]):
        idx = indices[i].item()
        if idx < 0:
            continue
        a = wa_t_all[idx, 0].float()  # [rank, d_model]
        b = wb_t_all[idx, 0].float()  # [out_features, rank]
        h = (x[i].float() @ a.T) * scale  # scale applied once, in shrink -- matches real vLLM add_lora()
        y_ref[i] += h @ b.T
    return y_ref


def run_case(token_lora_indices, label):
    torch.manual_seed(0)
    total_tokens = len(token_lora_indices)
    wa_t_all = (torch.randn(NUM_LORAS, 1, RANK, D_MODEL, device="cuda", dtype=torch.float16) * 0.02).contiguous()
    wb_t_all = (torch.randn(NUM_LORAS, 1, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16) * 0.02).contiguous()
    x = (torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    y_base = (torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    indices = torch.tensor(token_lora_indices, device="cuda", dtype=torch.int64)

    engine = WarpPipeEngine()
    y = y_base.clone()
    built = engine.build_seg_bounds(indices)
    assert built, "build_seg_bounds should only fail if total_tokens > WP_MAX_SEGMENTS, not the case here"
    handled = engine.add_lora_packed(y, x, wa_t_all, wb_t_all, ALPHA_SCALE)
    torch.cuda.synchronize()

    if not handled:
        print(f"{label}: NOT HANDLED (expected fallback)")
        return None

    y_ref = reference_add_lora(y_base, x, wa_t_all, wb_t_all, ALPHA_SCALE, indices)
    err = (y.float() - y_ref).abs().max().item()
    print(f"{label}: handled=True  max_abs_err={err:.5f}  {'PASS' if err < 0.05 else 'FAIL'}")
    return err


def main():
    results = []

    # Mixed runs including -1 (no adapter), matching real decode-batch shape.
    results.append(run_case([0, 0, 0, 0, -1, -1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], "mixed runs + no-adapter rows"))

    # Single adapter, exactly at the 32-token grid limit.
    results.append(run_case([3] * 32, "single run at WP_MAX_SEGMENT_TOKENS=32"))

    # All no-adapter (vacuous correctness).
    results.append(run_case([-1] * 8, "all no-adapter"))

    # Run exceeding the 32-token grid limit: the on-device segment-boundary
    # builder (build_seg_bounds_kernel, warp_pipe_r32.cu) splits this into
    # multiple <=32-token segments rather than rejecting it -- there's no
    # per-call Python-level fallback decision left once a call is part of a
    # captured CUDA graph, so splitting (not rejecting) is what makes the
    # long-run case usable at all under graph replay. Must still be
    # handled=True and numerically correct across the split boundary.
    err = run_case([5] * 33, "single run EXCEEDS 32 tokens (must split, not reject)")
    assert err is not None, "exceeding WP_MAX_SEGMENT_TOKENS must now be handled via splitting, not a fallback"
    results.append(err)

    # Many distinct single-token adapters back-to-back -- exercises the
    # split/segment-count bookkeeping at the other extreme (many small runs
    # instead of one long one). Also exercises DSAK's small-segment
    # (warp-per-token shuffle) path, since every run here is 1 token <=
    # WP_SMALL_SEG_THRESH=8.
    results.append(run_case(list(range(8)) * 4, "many short runs (worst case for segment count)"))

    # 8-token and 9-token single runs: WP_SMALL_SEG_THRESH is now 0 (DSAK's
    # wps path measured as a regression, see warp_pipe_metadata.h's comment
    # -- disabled, not deleted), so both go through the pipeline path. Kept
    # as regression coverage in case WP_SMALL_SEG_THRESH is ever raised again.
    results.append(run_case([4] * 8, "single 8-token run (pipeline path; was wps path pre-regression-fix)"))
    results.append(run_case([4] * 9, "single 9-token run (pipeline path)"))

    assert all(e is None or e < 0.05 for e in results), "some case failed"
    test_segment_reuse_across_layers()
    print("ALL PASS")


def test_segment_reuse_across_layers():
    """The core property: build_seg_bounds() runs ONCE per step, then
    multiple add_lora_packed() calls (one per LoRA layer in a real decode
    step) reuse those SAME boundaries against DIFFERENT wa_t_all/wb_t_all --
    mirroring a 13B model's ~80 add_lora() calls per step all sharing one
    boundary build. Verifies boundaries genuinely persist correctly across
    calls and each call's own A/B pointers are independently resolved (no
    cross-layer pointer staleness)."""
    torch.manual_seed(1)
    token_lora_indices = [0, 0, 0, -1, 1, 1, 1, 1, 1, 2, 2]
    total_tokens = len(token_lora_indices)
    indices = torch.tensor(token_lora_indices, device="cuda", dtype=torch.int64)
    x = (torch.randn(total_tokens, D_MODEL, device="cuda", dtype=torch.float16) * 0.1).contiguous()

    engine = WarpPipeEngine()
    assert engine.build_seg_bounds(indices)

    for layer in range(4):  # simulate 4 distinct LoRA layers in one step
        wa_t_all = (torch.randn(NUM_LORAS, 1, RANK, D_MODEL, device="cuda", dtype=torch.float16)
                    * 0.02).contiguous()
        wb_t_all = (torch.randn(NUM_LORAS, 1, OUT_FEATURES, RANK, device="cuda", dtype=torch.float16)
                    * 0.02).contiguous()
        y_base = (torch.randn(total_tokens, OUT_FEATURES, device="cuda", dtype=torch.float16) * 0.1).contiguous()
        y = y_base.clone()
        handled = engine.add_lora_packed(y, x, wa_t_all, wb_t_all, ALPHA_SCALE)
        torch.cuda.synchronize()
        assert handled
        y_ref = reference_add_lora(y_base, x, wa_t_all, wb_t_all, ALPHA_SCALE, indices)
        err = (y.float() - y_ref).abs().max().item()
        print(f"reuse-across-layers layer={layer}: max_abs_err={err:.5f} {'PASS' if err < 0.05 else 'FAIL'}")
        assert err < 0.05, f"layer {layer} got wrong result reusing step-global segment boundaries"


if __name__ == "__main__":
    main()
