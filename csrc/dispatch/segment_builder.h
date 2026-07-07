#pragma once

#include <vector>

#include "../bridge/warp_pipe_metadata.h"
#include "../memory/adapter_store.h"
#include "../memory/staging_buffer.h"

// Converts CASH-style sorted-batch output into the SegmentDescriptor[] array
// the kernels read. `seg_offsets` has K+1 entries (CASH segment boundaries);
// `sorted_adapter_ids` has K entries, one adapter id per segment.
void build_segment_table(WarpPipeMetadata* meta_out, const std::vector<int32_t>& sorted_adapter_ids,
                          const std::vector<int32_t>& seg_offsets, const AdapterStore& store,
                          StagingBufferPool* staging);
