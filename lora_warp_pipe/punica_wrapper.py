"""Wires WarpPipe into vLLM's single-slice LoRA path, PunicaWrapper.add_lora(),
which is what the o_proj/down_proj-shaped (row-parallel) LoRA layers call.

This is the only module here that needs AdapterSlots (the multi-LoRA serving system
this builds on): it subclasses its FusedPunicaWrapper. Nothing else in the package
imports it, so the rest of lora_warp_pipe works without AdapterSlots installed.

Why add_lora() and not add_lora_packed_nslice() (the method
FusedPunicaWrapper in fused_punica_wrapper.py already overrides for
QKV/gate_up's multi-slice packed layers): WarpPipe's kernel computes one
shrink + one expand for a single (A, B) pair, exactly add_lora()'s shape.
add_lora_packed_nslice handles MULTIPLE stacked slices sharing one input
(2-3 of them) -- adapting WarpPipe to that shape would mean looping it
per-slice (no real fusion win over what FusedPunicaWrapper already does) or
a deeper kernel rework. Out of scope here; this file only touches the
single-slice path FusedPunicaWrapper does NOT already cover.

Why this kernel call rebuilds segments from wa_t_all/wb_t_all fresh every
call, never caching adapter pointers by id: vLLM's LRUCacheWorkerLoRAManager
can reassign which physical adapter occupies a given lora_idx slot between
requests (eviction reusing slot indices). wa_t_all/wb_t_all are the SAME
persistently-allocated stacked tensors every call; only their CONTENTS change
across reassignments. add_lora_packed() (the new C++ method this file calls)
reads pointers straight from those tensors every call -- the same pattern
vLLM's own bgmv kernels already use -- so a slot reassignment between calls
can never produce a stale pointer.

Gated by a NEW env var, AS_WARP_PIPE (distinct from AS_FUSED_KERNEL and
AS_FUSED_PACKED_NSLICE -- this overrides a different method, so it can
coexist with FusedPunicaWrapper's existing add_lora_packed_nslice override
rather than conflict with it). Falls back to the stock add_lora() whenever:
prefill (this kernel is decode-shaped, like FusedPunicaWrapper's own
prefill exclusion), a sliced-output call (y_offset/y_slice_size set -- not
implemented here), rank != 32 (only rank this kernel build supports), the
WarpPipe extension isn't importable, or add_lora_packed() itself returns
False (rank != 32 again, the only thing it independently checks now -- see
below). A contiguous same-adapter run longer than WP_MAX_SEGMENT_TOKENS=32
is not a fallback case: build_seg_bounds_kernel (warp_pipe_r32.cu, called
once per step from update_metadata() below, not from add_lora()) splits it
into multiple <=32-token segments on-device instead, since CUDA graph
capture bakes in one path permanently -- there's no per-call decision left to
make once a call is part of a captured graph, so "fall back sometimes" was
never viable for that case. Every remaining fallback is always-correct, never
a crash and never silently wrong, mirroring AdapterSlots' own pattern in
fused_lora_layers.py and fused_punica_wrapper.py.

Segment boundaries are built once per decode step, not on
every add_lora() call. update_metadata() below builds them once per
decode step (it runs once per step, always before any add_lora() calls for
that step -- verified against vllm/worker/model_runner.py's real
execute_model()); add_lora_packed() now only resolves the calling layer's own
A_ptr/B_ptr from wa_t_all/wb_t_all and dispatches the kernels (warp-per-token
shuffle-broadcast path for segments <= WP_SMALL_SEG_THRESH tokens, the cp.async
pipeline otherwise) -- see
warp_pipe_bindings.cpp's build_seg_bounds()/add_lora_packed() docstrings.

Installation: same __class__-reassignment technique as
install_fused_punica_wrapper, and stacks on top of it cleanly --
FusedPunicaWrapperWarpPipe subclasses FusedPunicaWrapper, so installing it
preserves FusedPunicaWrapper's own add_lora_packed_nslice override
regardless of whether install_fused_punica_wrapper ran first.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from adapter_slots.kernel.fused_punica_wrapper import FusedPunicaWrapper, _VLLM_PUNICA_AVAILABLE

try:
    from .engine import WarpPipeEngine

    _WARP_PIPE_IMPORT_OK = True
except ImportError:  # extension not built; every entry point below falls back
    _WARP_PIPE_IMPORT_OK = False
    WarpPipeEngine = None  # type: ignore[assignment,misc]

_engine: Optional["WarpPipeEngine"] = None
_diagnostic_call_count = 0
_diagnostic_handled_count = 0


def _get_engine():
    """Lazy singleton: WarpPipeEngine.__init__ does cudaHostAlloc, which
    needs a live CUDA context -- must not run at import time (module import
    can happen before CUDA init, e.g. during argument parsing)."""
    global _engine
    if _engine is None:
        _engine = WarpPipeEngine()
    return _engine


def warp_pipe_available() -> bool:
    return _WARP_PIPE_IMPORT_OK and os.environ.get("AS_WARP_PIPE", "0") == "1"


class FusedPunicaWrapperWarpPipe(FusedPunicaWrapper):
    """Adds a WarpPipe-backed add_lora() on top of FusedPunicaWrapper's
    existing add_lora_packed_nslice() override.

    Also overrides update_metadata():
    builds the segment-BOUNDARY table (adapter_id/token_start/token_count)
    once per decode step here, instead of once per add_lora() call the way
    the previous version of this file did. update_metadata() is called by
    vllm/worker/model_runner.py's execute_model() at the very top, in plain
    eager Python, before the graph-replay-vs-eager branch is even decided --
    verified directly against that source -- so it is always-uncaptured on
    every step, including the one-time capture pass itself. A 13B model's
    decode step calls add_lora() ~80 times (once per LoRA-enabled linear
    layer) with IDENTICAL segment boundaries every time (only each layer's
    own wa_t_all/wb_t_all differ) -- rebuilding those boundaries on every one
    of those 80 calls, as the previous design did, was redundant 79 times out
    of 80.
    """

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size,
                         long_lora_context=None) -> None:
        super().update_metadata(mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size,
                                 long_lora_context)
        if not self.is_prefill and warp_pipe_available():
            _get_engine().build_seg_bounds(self.token_lora_indices)

    def add_lora(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        wa_t_all: torch.Tensor,
        wb_t_all: torch.Tensor,
        scale: float,
        y_offset: Optional[int] = None,
        y_slice_size: Optional[int] = None,
        *,
        buffer: Optional[torch.Tensor] = None,
    ) -> None:
        # No torch.cuda.is_current_stream_capturing() guard here: segment
        # boundaries are built on-device with zero host syncs (in
        # update_metadata() above, not here -- see that method's docstring),
        # and the dsak shrink/expand kernels dispatched below use fixed
        # launch grids and read meta's live on-device contents on every
        # replay rather than baking in whatever was true at capture time --
        # verified directly via an explicit torch.cuda.graph()
        # capture-with-one-set-of-indices / replay-with-a-different-set test
        # (test_warp_pipe_graph_safety.py), not just "didn't crash."
        if (
            self.is_prefill
            or y_offset is not None
            or y_slice_size is not None
            or not warp_pipe_available()
            or not x.is_cuda
            or wb_t_all.size(-1) != 32
        ):
            return super().add_lora(y, x, wa_t_all, wb_t_all, scale, y_offset, y_slice_size, buffer=buffer)

        handled = _get_engine().add_lora_packed(y, x, wa_t_all, wb_t_all, float(scale))
        if not handled:
            return super().add_lora(y, x, wa_t_all, wb_t_all, scale, y_offset, y_slice_size, buffer=buffer)


class DummyWarpPipeWrapper(FusedPunicaWrapper):
    """Diagnostic-only ablation: identical to FusedPunicaWrapper in every
    behavior (no overrides at all) -- exists solely to test whether the
    __class__ reassignment mechanism itself (not anything WarpPipe's
    overrides actually do) is responsible for the C7-vs-C8 live throughput
    regression. See install_warp_pipe_dummy() below. Remove once that
    question is answered."""


def install_warp_pipe_dummy(lora_manager) -> bool:
    """Same __class__-reassignment mechanism as install_warp_pipe_add_lora(),
    but onto a subclass with zero behavioral overrides. If this alone
    reproduces the regression, the cause is something in vLLM that
    dispatches on type(wrapper)/isinstance(wrapper, ...), not anything in
    WarpPipe's actual add_lora()/update_metadata() code or kernels."""
    if not _VLLM_PUNICA_AVAILABLE or lora_manager is None:
        return False
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    if adapter_manager is None:
        return False
    wrapper = getattr(adapter_manager, "punica_wrapper", None)
    if wrapper is None or isinstance(wrapper, DummyWarpPipeWrapper):
        return False
    wrapper.__class__ = DummyWarpPipeWrapper
    return True


class MetadataOnlyWarpPipe(FusedPunicaWrapper):
    """Diagnostic-only ablation: overrides update_metadata() exactly like
    FusedPunicaWrapperWarpPipe (the real eager, per-step, uncaptured
    build_seg_bounds() kernel launch) but does NOT override add_lora() --
    every add_lora() call falls back to the stock path, so no DSAK kernels
    ever run. Isolates whether the regression comes from interleaving one
    eager kernel launch into the otherwise all-graph-replay decode loop,
    independent of WarpPipe's actual LoRA kernels. Remove once answered."""

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size,
                         long_lora_context=None) -> None:
        super().update_metadata(mapping, lora_index_to_id, max_loras, vocab_size, extra_vocab_size,
                                 long_lora_context)
        if not self.is_prefill and warp_pipe_available():
            _get_engine().build_seg_bounds(self.token_lora_indices)


def install_warp_pipe_metadata_only(lora_manager) -> bool:
    if not _VLLM_PUNICA_AVAILABLE or lora_manager is None or not _WARP_PIPE_IMPORT_OK:
        return False
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    if adapter_manager is None:
        return False
    wrapper = getattr(adapter_manager, "punica_wrapper", None)
    if wrapper is None or isinstance(wrapper, MetadataOnlyWarpPipe):
        return False
    wrapper.__class__ = MetadataOnlyWarpPipe
    return True


def install_warp_pipe_add_lora(lora_manager) -> bool:
    """Reassign the shared PunicaWrapper instance's __class__ to
    FusedPunicaWrapperWarpPipe. Safe to call whether or not
    install_fused_punica_wrapper() ran first (see module docstring) -- the
    resulting class always has both overrides either way.

    Returns False (no-op, not an error) if vLLM LoRA support is unavailable,
    lora_manager is None, the WarpPipe extension failed to import, or the
    wrapper is already a FusedPunicaWrapperWarpPipe.
    """
    if not _VLLM_PUNICA_AVAILABLE or lora_manager is None or not _WARP_PIPE_IMPORT_OK:
        return False
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    if adapter_manager is None:
        return False
    wrapper = getattr(adapter_manager, "punica_wrapper", None)
    if wrapper is None or isinstance(wrapper, FusedPunicaWrapperWarpPipe):
        return False
    wrapper.__class__ = FusedPunicaWrapperWarpPipe
    return True
