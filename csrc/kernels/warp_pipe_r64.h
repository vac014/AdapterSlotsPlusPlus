#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../bridge/warp_pipe_metadata.h"

// rank=64 cp.async pipeline shrink/expand, generalized from
// warp_pipe_r32.cu's hand-specialized kernels via warp_pipe_rgen.cuh's
// template<int RANK>. No DSAK/build_segments/build_seg_bounds variant here --
// see warp_pipe_rgen.cuh's header comment for why (DSAK is RANK==32-only and
// unreachable in the live path anyway; build_segments() at the Python
// binding layer goes through segment_builder.cpp's build_segment_table,
// already rank-agnostic, not through a per-rank kernel).
void launch_bgmv_shrink_r64(const half* X, half* H, const WarpPipeMetadata* meta, int d_model,
                               cudaStream_t stream);

void launch_bgmv_expand_r64(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features,
                               cudaStream_t stream);
