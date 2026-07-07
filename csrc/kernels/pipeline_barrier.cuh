#pragma once

// These are full __syncthreads() rather than named barriers: correct across all
// warp counts, and it sidesteps the 16-named-barriers-per-kernel limit on SM86.
// Worth upgrading to real named barriers only if profiling shows
// smsp__warp_issue_stalled_wait_pct above 15% at these sync points.

__device__ __forceinline__ void wp_barrier_tile_ready() {
  __syncthreads();
}

__device__ __forceinline__ void wp_barrier_tile_consumed() {
  __syncthreads();
}
