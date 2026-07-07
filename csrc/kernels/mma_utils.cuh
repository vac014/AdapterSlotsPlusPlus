#pragma once

#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Why per-lane FMA and not tensor-core mma.sync.
//
// In the regime this kernel targets (K<=16, T<=16, decode), the work is firmly
// bandwidth-bound: tensor-core compute for one tile is on the order of 3.4 ns,
// negligible beside the tile-load latency it would be hiding behind. Raw
// `mma.sync.aligned.m16n8k16` + `ldmatrix.sync.aligned.m8n8.x4` would buy
// nothing here, and a hand-derived fragment layout is an easy way to get
// silently wrong answers.
//
// So each lane computes its own dot product with plain FMAs and owns one full
// output element (the same output-ownership pattern as warp_pipe_r32.cu).
// There is no cross-lane shuffle-reduce at all, and no scalar lane-0 write
// wasting 31/32 of the warp's write bandwidth.
//
// Worth revisiting only if profiling a prefill-like (T>=16) workload shows
// this kernel is genuinely compute-bound, and at that point CUTLASS
// GemmUniversal is the right tool rather than a hand-rolled mma.sync pipe.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float lane_dot_fp16(const half* a, const half* b, int len) {
  float acc = 0.0f;
  // TRIED AND REVERTED: bounding this to "#pragma unroll 8" cut shrink's
  // register usage 165->47/thread and raised its occupancy ceiling
  // 16.67%->66.67% (fewer launch waves, 24.4->6.1) -- exactly the fix the
  // register-pressure diagnosis predicted. Measured wall-clock result was
  // WORSE anyway (K=16,T=16 speedup vs vLLM: 0.954x -> 0.678x), so whatever
  // is actually slow at high K,T is not occupancy-ceiling-bound the way the
  // isolated metric suggested. Reverted to full unroll pending a different
  // diagnosis.
#pragma unroll
  for (int i = 0; i < len; i++) {
    acc += __half2float(a[i]) * __half2float(b[i]);
  }
  return acc;
}
