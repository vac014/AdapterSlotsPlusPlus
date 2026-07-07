#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../bridge/warp_pipe_metadata.h"

// rank=32 only. warp_pipe_r8/r16/r64.cu carry the other rank variants; this
// dispatcher does not route to them yet.
bool warp_pipe_rank_supported(int rank);

void dispatch_build_segments(const int64_t* token_lora_indices, int total_tokens,
                              const half* wa_base, int64_t wa_stride0,
                              const half* wb_base, int64_t wb_stride0,
                              float alpha_scale, int rank, int d_model, int out_features,
                              WarpPipeMetadata* meta, cudaStream_t stream);

void dispatch_bgmv_shrink(const half* X, half* H, const WarpPipeMetadata* meta, int d_model, int rank,
                           cudaStream_t stream);

void dispatch_bgmv_expand(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features, int rank,
                           cudaStream_t stream);

// Once-per-step boundary build (see warp_pipe_r32.h's
// launch_build_seg_bounds_r32).
void dispatch_build_seg_bounds(const int64_t* token_lora_indices, int total_tokens, WarpPipeMetadata* meta,
                                cudaStream_t stream);

// DSAK live-path shrink/expand (see warp_pipe_r32.h's
// launch_bgmv_shrink_dsak_r32/launch_bgmv_expand_dsak_r32).
void dispatch_bgmv_shrink_dsak(const half* X, half* H, const WarpPipeMetadata* meta, const half* wa_base,
                                int64_t wa_stride0, float alpha_scale, int d_model, int rank, cudaStream_t stream);

void dispatch_bgmv_expand_dsak(const half* H, half* Y, const WarpPipeMetadata* meta, const half* wb_base,
                                int64_t wb_stride0, int out_features, int rank, cudaStream_t stream);
