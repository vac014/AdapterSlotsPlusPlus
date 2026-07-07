// Prefetch copy kernel: pulls segment k+1's adapter weights in behind segment
// k's compute.
//
// A hand-rolled kernel rather than cudaMemcpyAsync, for one concrete reason: cudaMemcpyAsync's throughput for
// device-to-device copies of this size (tens to low hundreds of KB) is
// dominated by its own launch/dispatch overhead at these sizes on this
// driver, and a simple grid-strided float4 copy kernel gives direct control
// over how many threads/blocks service one copy -- sized here to finish
// well within one tile-load's worth of time on the prefetch stream, which is
// the whole point (hide segment k+1's cold-start load behind segment k's
// compute). Vectorized as float4 (16B/thread/iteration) since both A and B
// buffers are always 16-byte aligned (AdapterStore::register_adapter
// enforces this already).
#include <cuda_runtime.h>

#include "../bridge/warp_pipe_metadata.h"

// GPU-side bridge readback, for test_scheduler_bridge.py. No kernel here reads
// bridge_ptr for real yet (GWAR/Whittle-driven in-kernel decisions are not
// built), so this exists to prove the device pointer
// WarpPipeMetadata.bridge_ptr is actually readable from a real kernel
// launch, not just from host-pinned memory (which read_bridge_debug()
// already proves separately, but that's a CPU-side check, not a GPU one).
__global__ void bridge_readback_kernel(const WarpPipeMetadata* meta, float* out_whittle, uint32_t* out_scalars) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const SchedulerKernelBridge* b = meta->bridge_ptr;
    for (uint32_t i = 0; i < b->num_active_adapters; i++) {
      out_whittle[i] = b->whittle_scores[i];
    }
    out_scalars[0] = b->num_active_adapters;
    out_scalars[1] = b->num_segments;
    out_scalars[2] = b->step_id;
  }
}

void launch_bridge_readback(const WarpPipeMetadata* meta, float* out_whittle, uint32_t* out_scalars,
                             cudaStream_t stream) {
  bridge_readback_kernel<<<1, 32, 0, stream>>>(meta, out_whittle, out_scalars);
}

__global__ void prefetch_copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, int num_float4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (; i < num_float4; i += stride) {
    dst[i] = src[i];
  }
}

void launch_prefetch_copy(const void* src, void* dst, size_t num_bytes, cudaStream_t stream) {
  // num_bytes is always rank(32) * d_model * 2 (fp16) or out_features * rank(32) * 2 --
  // always a multiple of 16 bytes for any d_model/out_features used in this
  // project (4096, 5120, ... all divisible by 8 fp16 elements).
  const int num_float4 = static_cast<int>(num_bytes / sizeof(float4));
  const int threads = 256;
  const int blocks = (num_float4 + threads - 1) / threads;
  prefetch_copy_kernel<<<max(blocks, 1), threads, 0, stream>>>(reinterpret_cast<const float4*>(src),
                                                                reinterpret_cast<float4*>(dst), num_float4);
}
