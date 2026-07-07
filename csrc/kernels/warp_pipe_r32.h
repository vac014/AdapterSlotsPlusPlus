#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../bridge/warp_pipe_metadata.h"

// Builds meta->segments[]/num_segments entirely on-device from
// token_lora_indices (a GPU int64 tensor) -- no host sync, fixed grid=(1,1,1).
// See warp_pipe_r32.cu's build_segments_kernel for why this exists and why
// it's safe to call mid-CUDA-graph-capture.
void launch_build_segments_r32(const int64_t* token_lora_indices, int total_tokens,
                                const half* wa_base, int64_t wa_stride0,
                                const half* wb_base, int64_t wb_stride0,
                                float alpha_scale, int rank, int d_model, int out_features,
                                WarpPipeMetadata* meta, cudaStream_t stream);

// Shrink: H[t, r] = alpha_scale * sum_d X[t, d] * A[r, d]
void launch_bgmv_shrink_r32(const half* X, half* H, const WarpPipeMetadata* meta,
                             int d_model, cudaStream_t stream);

// Expand: Y[t, o] += alpha_scale * sum_r H[t, r] * B[o, r]   (Y holds Y_base on entry)
void launch_bgmv_expand_r32(const half* H, half* Y, const WarpPipeMetadata* meta,
                             int out_features, cudaStream_t stream);

// Builds ONLY segment boundaries
// (adapter_id/token_start/token_count), no A_ptr/B_ptr/rank/d_model/
// out_features/alpha_scale -- meant to be called ONCE PER DECODE STEP from
// PunicaWrapper.update_metadata(), not once per layer like
// launch_build_segments_r32 above. See warp_pipe_r32.cu's
// build_seg_bounds_kernel comment.
void launch_build_seg_bounds_r32(const int64_t* token_lora_indices, int total_tokens,
                                  WarpPipeMetadata* meta, cudaStream_t stream);

// Live-path shrink/expand: reads only
// seg.adapter_id/token_start/token_count from meta (assumed already built by
// launch_build_seg_bounds_r32 this step); A_ptr/B_ptr are computed from
// wa_base/wb_base + seg.adapter_id*stride0 instead of meta's stale struct
// fields. Branches on-device between the cp.async pipeline path (segments >
// WP_SMALL_SEG_THRESH tokens) and a warp-per-token shuffle-broadcast path
// (segments <= WP_SMALL_SEG_THRESH) -- see warp_pipe_r32.cu.
void launch_bgmv_shrink_dsak_r32(const half* X, half* H, const WarpPipeMetadata* meta,
                                  const half* wa_base, int64_t wa_stride0, float alpha_scale,
                                  int d_model, cudaStream_t stream);

void launch_bgmv_expand_dsak_r32(const half* H, half* Y, const WarpPipeMetadata* meta,
                                  const half* wb_base, int64_t wb_stride0, int out_features,
                                  cudaStream_t stream);
