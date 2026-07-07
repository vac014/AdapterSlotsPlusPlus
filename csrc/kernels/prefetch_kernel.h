#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../bridge/warp_pipe_metadata.h"

void launch_prefetch_copy(const void* src, void* dst, size_t num_bytes, cudaStream_t stream);

void launch_bridge_readback(const WarpPipeMetadata* meta, float* out_whittle, uint32_t* out_scalars,
                             cudaStream_t stream);
