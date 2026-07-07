// Standalone CUDA binary smoke test for pipeline_barrier.cuh's barrier wrappers.
// They are __syncthreads() wrappers today, not real named barriers; this test
// exists to catch a regression if that ever changes (a real named-barrier
// implementation could silently scope to the wrong warp subset and this is
// the kind of bug that "didn't crash" hides completely without an explicit
// cross-warp-visibility check like the one below).
//
// Build: nvcc -gencode arch=compute_86,code=sm_86 -I../../csrc/kernels
//        test_barrier_sync.cu -o test_barrier_sync
#include <cstdio>

#include "pipeline_barrier.cuh"

constexpr int N_WARPS = 8;
constexpr int N_THREADS = N_WARPS * 32;

// Each warp writes its warp_id into shared memory, then after the barrier
// every thread reads ALL other warps' values -- this only produces correct
// (non-garbage) values for every slot if the barrier actually made every
// warp's write visible to every other warp before any read happens.
__global__ void barrier_visibility_kernel(int* out_mismatches) {
  __shared__ int slot[N_WARPS];
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  if (lane_id == 0) {
    slot[warp_id] = warp_id * 1000 + 7;  // distinctive, easy to spot corruption
  }
  wp_barrier_tile_ready();

  int local_mismatches = 0;
  for (int w = 0; w < N_WARPS; w++) {
    if (slot[w] != w * 1000 + 7) local_mismatches++;
  }
  wp_barrier_tile_consumed();

  // Reduce mismatches across all threads via atomics (small N, fine for a test).
  if (local_mismatches > 0) {
    atomicAdd(out_mismatches, local_mismatches);
  }
}

int main() {
  int* d_mismatches;
  cudaMalloc(&d_mismatches, sizeof(int));
  cudaMemset(d_mismatches, 0, sizeof(int));

  barrier_visibility_kernel<<<64, N_THREADS>>>(d_mismatches);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("FAIL: kernel launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  int h_mismatches = 0;
  cudaMemcpy(&h_mismatches, d_mismatches, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_mismatches);

  if (h_mismatches != 0) {
    printf("FAIL: %d cross-warp visibility mismatches after wp_barrier_tile_ready()\n", h_mismatches);
    return 1;
  }
  printf("PASS: wp_barrier_tile_ready()/wp_barrier_tile_consumed() give full cross-warp visibility "
         "(64 blocks x %d warps, zero mismatches)\n", N_WARPS);
  return 0;
}
