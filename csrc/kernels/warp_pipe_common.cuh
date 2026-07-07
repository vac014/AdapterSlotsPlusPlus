#pragma once

#include "../bridge/warp_pipe_metadata.h"

// Tokens processed per block (one "chunk" = grid.y slot). Kept small so a
// fixed, generous grid.y (WP_MAX_CHUNKS_PER_SEGMENT) covers any real segment
// length without per-call grid resizing (CUDA-graph safety -- see
// warp_pipe_metadata.h's note on WarpPipeMetadata).
static constexpr int WP_TOKEN_CHUNK = 4;
// WP_MAX_SEGMENT_TOKENS now lives in warp_pipe_metadata.h (host code building
// real segments needs it too, and that header is the one .cpp TUs can include
// without pulling in this file's __device__ code).
static constexpr int WP_MAX_CHUNKS_PER_SEGMENT =
    (WP_MAX_SEGMENT_TOKENS + WP_TOKEN_CHUNK - 1) / WP_TOKEN_CHUNK;  // ceil, = 8

// Rank-32 warp split: 8-warp block, 2 producer + 6 consumer.
//
// TRIED AND REVERTED: 6 doesn't divide evenly into WP_TOKEN_CHUNK (4) or
// TILE_O (128), leaving 2 of 6 consumer warps fully idle in both kernels
// (confirmed via ncu: sm__warps_active 16.6% shrink, 38-42% expand at
// K=16,T=16). Dropping to 4 consumer warps (NUM_WARPS=6) eliminated the idle
// warps but measured WORSE overall (mean speedup vs vLLM: 1.528x -> 1.309x;
// K=16,T=16 specifically: 0.954x -> 0.803x) -- fewer total warps per block
// apparently hurt latency-hiding more than the idle-warp fix helped. Kept at
// 6 consumer warps; the idle-warp issue needs a different fix (e.g. tile
// sizing) that doesn't reduce total warp count.
static constexpr int WP_RANK32_PRODUCER_WARPS = 2;
static constexpr int WP_RANK32_CONSUMER_WARPS = 6;
static constexpr int WP_RANK32_NUM_WARPS      = WP_RANK32_PRODUCER_WARPS + WP_RANK32_CONSUMER_WARPS;

__device__ __forceinline__ bool wp_is_producer(int warp_id, int producer_warps) {
  return warp_id < producer_warps;
}
