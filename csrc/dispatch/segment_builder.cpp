#include "segment_builder.h"

#include <stdexcept>

void build_segment_table(WarpPipeMetadata* meta_out, const std::vector<int32_t>& sorted_adapter_ids,
                          const std::vector<int32_t>& seg_offsets, const AdapterStore& store,
                          StagingBufferPool* staging) {
  const int K = static_cast<int>(sorted_adapter_ids.size());
  if (static_cast<int>(seg_offsets.size()) != K + 1) {
    throw std::runtime_error("build_segment_table: seg_offsets must have K+1 entries");
  }
  if (K > WP_MAX_SEGMENTS) {
    throw std::runtime_error("build_segment_table: K exceeds WP_MAX_SEGMENTS");
  }

  for (int i = 0; i < K; i++) {
    const int32_t adapter_id = sorted_adapter_ids[i];
    const AdapterStore::Entry& e = store.lookup(adapter_id);
    SegmentDescriptor& sd = meta_out->segments[i];
    sd.adapter_id = adapter_id;
    sd.token_start = seg_offsets[i];
    sd.token_count = seg_offsets[i + 1] - seg_offsets[i];
    sd.rank = e.rank;
    sd.d_model = e.d_model;
    sd.out_features = e.out_features;
    sd.alpha_scale = e.alpha_scale;
    sd.promo_eligible = 0;
    sd.burst_active = 0;
    sd.use_staging = 0;
    sd.prefetch_ready = 0;
    sd.staging_A_ptr = 0;
    sd.staging_B_ptr = 0;
    // Invariant: use_staging=1 and a null staging_A_ptr/staging_B_ptr is a
    // null-pointer read inside the kernel. The two are set together, below, or
    // not at all.
    int slot = staging != nullptr ? staging->find_slot_for_adapter(adapter_id) : -1;
    if (slot >= 0 && staging->is_ready(slot)) {
      sd.use_staging = 1;
      sd.prefetch_ready = 1;
      sd.staging_A_ptr = reinterpret_cast<uintptr_t>(staging->a_ptr(slot));
      sd.staging_B_ptr = reinterpret_cast<uintptr_t>(staging->b_ptr(slot));
    }
    sd.A_ptr = e.A_ptr;
    sd.B_ptr = e.B_ptr;
  }
  // GRAPH-SAFETY-CRITICAL: this is the only field that may legitimately vary
  // the kernel's effective work between calls; grid/block/launch shape never
  // change (see warp_pipe_metadata.h).
  meta_out->num_segments = static_cast<uint32_t>(K);
}
