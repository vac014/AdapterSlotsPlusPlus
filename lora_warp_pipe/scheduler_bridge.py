"""Carries AdapterSlots scheduler state into the WarpPipe engine once per tick.

AdapterSlots is the multi-LoRA serving system this builds on (see README). The
scheduler objects are duck-typed, so nothing here imports it: pass in anything with
the shape of adapter_slots.dispatch.whittle.WhittleDispatcher,
adapter_slots.control.estimator.ArrivalRateEstimator,
adapter_slots.kernel.wgkp_dispatcher.WGKPDispatcher and
adapter_slots.metrics.gwar.compute_gwar.

Two adaptations, both load-bearing:

  1. AdapterSlots identifies adapters by STRING id (Dict[str, float] throughout
     whittle.py and estimator.py); the kernel's structs use fixed int32 slots
     (WP_MAX_ADAPTERS=64). AdapterIdMap below is the translation layer, and it must
     be the SAME instance used wherever engine.register_adapter() was called with int
     ids, so the two id spaces cannot drift apart.

  2. compute_gwar() is a point-in-time O(N) scan; there is no temporal "GWAR over the
     next 3 steps" upstream to borrow. GwarHistory below is a minimal EWMA over
     successive readings, enough to fill the bridge's gwar_pred_next3 field with
     something better than a placeholder. It is not a validated predictor, and it is
     not claimed to be one.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from .engine import WarpPipeEngine


class AdapterIdMap:
    """Bijection between AS++'s string adapter_ids and the small int32 ids
    LoRA-WarpPipe's fixed-size structs use. One instance must be shared
    across every register_adapter()/build_segments()/write_bridge() call
    site for a given engine -- ids are assigned in first-seen order and
    never reused within a process lifetime."""

    def __init__(self):
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: List[str] = []

    def get_or_assign(self, adapter_id: str) -> int:
        if adapter_id not in self._str_to_int:
            new_id = len(self._int_to_str)
            self._str_to_int[adapter_id] = new_id
            self._int_to_str.append(adapter_id)
        return self._str_to_int[adapter_id]

    def to_int(self, adapter_id: str) -> int:
        return self._str_to_int[adapter_id]

    def ordered_ids(self) -> List[str]:
        """All known adapter ids, in their assigned int-index order -- index i
        of this list corresponds to int id i everywhere else in this module."""
        return list(self._int_to_str)

    def __len__(self) -> int:
        return len(self._int_to_str)


class GwarHistory:
    """EWMA smoother over compute_gwar() readings, used to fill the bridge's
    gwar_pred_next3 field. See module docstring point 2 -- this is a minimal
    addition, not an existing AS++ predictor."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._ewma: Dict[str, float] = {}

    def update(self, adapter_id: str, gwar_value: float) -> float:
        prev = self._ewma.get(adapter_id, gwar_value)
        new = self.alpha * gwar_value + (1.0 - self.alpha) * prev
        self._ewma[adapter_id] = new
        return new

    def get(self, adapter_id: str) -> float:
        return self._ewma.get(adapter_id, 0.0)


def segments_to_warp_pipe(segments, id_map: AdapterIdMap):
    """Converts a List[SegmentDescriptor] from
    WGKPDispatcher.segment_and_promote()
    into the (sorted_adapter_ids, seg_offsets) arrays engine.build_segments()
    expects (the same layout csrc/dispatch/segment_builder.cpp consumes).

    Relies on the same AlignmentBuffer invariant WGKPDispatcher itself relies
    on: tokens are already adapter-sorted, so each SegmentDescriptor's
    segment_size is the row-count of a CONTIGUOUS run in whatever tensor the
    model runner built in that same dispatch order -- token_start/token_count
    are therefore just the cumulative sum of segment_size, not seg.seq_ids
    (those are vLLM sequence ids, unrelated to physical row offsets).
    """
    sorted_adapter_ids: List[int] = []
    seg_offsets: List[int] = [0]
    cum = 0
    for seg in segments:
        sorted_adapter_ids.append(id_map.get_or_assign(seg.adapter_id))
        cum += seg.segment_size
        seg_offsets.append(cum)
    return sorted_adapter_ids, seg_offsets


class SchedulerBridge:
    """Pulls real AS++ scheduler state into the WarpPipeEngine's
    SchedulerKernelBridge once per scheduler tick.

    Construct once per engine lifetime; call write() once per tick with
    whatever subset of the real scheduler objects are active for the current
    AS_MODE (whittle_dispatcher/estimator/mwc are all Optional because not
    every AS++ mode instantiates all of them -- see
    adapter_slots.integrations.vllm_scheduler's AlignmentAwareScheduler).
    """

    def __init__(self, engine: WarpPipeEngine, id_map: AdapterIdMap):
        self._engine = engine
        self._id_map = id_map
        self._gwar_history = GwarHistory()
        self._step_id = 0

    def write(
        self,
        num_segments: int,
        whittle_dispatcher=None,
        fill_fracs: Optional[Dict[str, float]] = None,
        lambda_est: Optional[Dict[str, float]] = None,
        estimator=None,
        gwar_readings: Optional[Dict[str, float]] = None,
        mwc=None,
        global_load: float = 0.0,
    ) -> None:
        ordered = self._id_map.ordered_ids()
        n = len(ordered)
        if n == 0:
            self._step_id += 1
            return

        whittle_scores = np.zeros(n, dtype=np.float32)
        if whittle_dispatcher is not None and fill_fracs is not None and lambda_est is not None:
            indices = whittle_dispatcher.compute_indices(fill_fracs, lambda_est)
            for i, aid in enumerate(ordered):
                whittle_scores[i] = indices.get(aid, 0.0)

        lambda_hat = np.zeros(n, dtype=np.float32)
        if estimator is not None:
            for i, aid in enumerate(ordered):
                lambda_hat[i] = estimator.get_rate(aid)

        gwar_pred_next3 = np.zeros(n, dtype=np.float32)
        if gwar_readings is not None:
            for i, aid in enumerate(ordered):
                if aid in gwar_readings:
                    self._gwar_history.update(aid, gwar_readings[aid])
                gwar_pred_next3[i] = self._gwar_history.get(aid)

        promo_eligible = np.zeros(n, dtype=np.uint8)
        if mwc is not None:
            for i, aid in enumerate(ordered):
                if mwc.is_merged(aid):
                    promo_eligible[i] = 1

        t_remaining_ms = np.zeros(n, dtype=np.float32)  # no prefetch-time estimate is wired yet
        burst_active = np.zeros(n, dtype=np.uint8)  # no burst-epoch tracker wired yet
        is_hot = np.zeros(n, dtype=np.uint8)
        tile_size_code = np.zeros(n, dtype=np.uint8)
        tile_size_code[gwar_pred_next3 > 0.6] = 1
        tile_size_code[gwar_pred_next3 > 0.85] = 2

        self._engine.write_bridge(
            torch.from_numpy(whittle_scores),
            torch.from_numpy(t_remaining_ms),
            torch.from_numpy(gwar_pred_next3),
            torch.from_numpy(lambda_hat),
            torch.from_numpy(burst_active),
            torch.from_numpy(promo_eligible),
            torch.from_numpy(is_hot),
            torch.from_numpy(tile_size_code),
            n,
            num_segments,
            self._step_id,
            global_load,
        )
        self._step_id += 1
