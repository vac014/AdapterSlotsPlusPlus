import torch  # noqa: F401 -- must load before warp_pipe_ext: the extension links libc10/libtorch but has no rpath to them; torch's own import puts them in the process (RTLD_GLOBAL) so the dynamic linker can resolve warp_pipe_ext's undefined symbols.
import warp_pipe_ext

from .config import WarpPipeConfig


class WarpPipeEngine:
    def __init__(self, config: WarpPipeConfig = None):
        self.config = config or WarpPipeConfig()
        self._ext = warp_pipe_ext.WarpPipeExtension()

    def register_adapter(self, adapter_id, A, B, alpha_scale, rank=None):
        rank = rank or self.config.rank
        self._ext.register_adapter(adapter_id, A, B, rank, alpha_scale)

    def evict_adapter(self, adapter_id):
        self._ext.evict_adapter(adapter_id)

    def build_segments(self, sorted_adapter_ids, seg_offsets):
        self._ext.build_segments(sorted_adapter_ids, seg_offsets)

    def shrink(self, X, d_model, rank, total_tokens):
        return self._ext.shrink(X, d_model, rank, total_tokens)

    def expand(self, H, Y, out_features, rank):
        self._ext.expand(H, Y, out_features, rank)

    def build_seg_bounds(self, token_lora_indices):
        """Builds segment boundaries once
        per decode step, ahead of however many add_lora_packed() calls follow
        for that step. Returns False only if total_tokens > WP_MAX_SEGMENTS."""
        return self._ext.build_seg_bounds(token_lora_indices)

    def read_bridge_debug(self):
        """Debug-only readback of the SchedulerKernelBridge, for
        test_metadata_bridge.py's write/read round-trip check."""
        return self._ext.read_bridge_debug()

    def read_bridge_via_kernel(self):
        """GPU-side readback via a real kernel dereferencing bridge_ptr, for
        test_scheduler_bridge.py. Returns [whittle_scores..., n, num_segments, step_id]."""
        return self._ext.read_bridge_via_kernel()

    def prefetch_adapter(self, adapter_id):
        """Issue an async copy of adapter_id's A/B weights into a staging slot, on
        the extension's own prefetch stream, concurrent with whatever the main
        stream is doing. Returns the slot index, or -1 if no slot was free."""
        return self._ext.prefetch_adapter(adapter_id)

    def is_prefetch_ready(self, adapter_id):
        return self._ext.is_prefetch_ready(adapter_id)

    def release_prefetch(self, adapter_id):
        self._ext.release_prefetch(adapter_id)

    def add_lora_packed(self, y, x, wa_t_all, wb_t_all, scale):
        """Live-serving entry point mirroring vLLM's PunicaWrapper.add_lora().
        Assumes build_seg_bounds() already ran this step for the indices this
        call's rows correspond to. Returns False if this kernel can't handle
        the call (caller must fall back to the stock path) -- see
        add_lora_packed's C++ docstring for exactly which cases that covers."""
        return self._ext.add_lora_packed(y, x, wa_t_all, wb_t_all, scale)

    def write_bridge(self, whittle_scores, t_remaining_ms, gwar_pred_next3, lambda_hat, burst_active,
                      promo_eligible, is_hot, tile_size_code, num_active_adapters, num_segments, step_id,
                      global_load):
        self._ext.write_bridge(whittle_scores, t_remaining_ms, gwar_pred_next3, lambda_hat, burst_active,
                                promo_eligible, is_hot, tile_size_code, num_active_adapters, num_segments, step_id,
                                global_load)
