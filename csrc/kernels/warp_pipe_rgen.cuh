#pragma once
// warp_pipe_rgen.cuh -- generalized rank shrink/expand for rank in {8,16,64}
// (rank=32 stays on warp_pipe_r32.cu's hand-specialized path, untouched --
// that path is the live-serving DSAK path and is not to be disturbed).
//
// DSAK (warp_pipe_r32.cu's bgmv_shrink_dsak_kernel/bgmv_expand_dsak_kernel)
// is NOT generalized here. DSAK's small-segment warp-per-token path
// (wps_shrink_warp/wps_expand_warp) exploits RANK==warpSize==32 directly --
// one lane per rank-output, one __shfl_sync broadcast per rank index -- and
// is unreachable in the live path anyway (WP_SMALL_SEG_THRESH=0, see
// warp_pipe_metadata.h). Only the cp.async pipeline kernels (the ones
// shrink()/expand()/build_segments() in warp_pipe_bindings.cpp actually
// exercise -- the scheduler-bridge/benchmark-harness path, not live serving)
// are generalized.
//
// Difference from warp_pipe_r32.cu's bgmv_shrink_kernel: r32 has each lane
// own EXACTLY one rank-output column because RANK==warpSize. Here RANK may
// be < 32 (8, 16: lanes >= RANK simply have no column to own and sit out the
// per-rank accumulation) or > 32 (64: each lane owns RANK/32 = 2 columns,
// looping r = lane_id, lane_id+32). bgmv_expand's lane_dot_fp16(b_row, h_row,
// RANK) already loops over a runtime-irrelevant, compile-time RANK-length
// dot product with no lane-count assumption at all -- it needed no change,
// just instantiation at this RANK.

#include "warp_pipe_r32.h"  // SegmentDescriptor/WarpPipeMetadata via warp_pipe_metadata.h, reused launch sig style
#include "warp_pipe_common.cuh"
#include "cp_async_utils.cuh"
#include "mma_utils.cuh"
#include "pipeline_barrier.cuh"

namespace wp_rgen {

constexpr int TILE_K = 256;
constexpr int TILE_O = 256;
constexpr int PRODUCER_WARPS = 2;
constexpr int CONSUMER_WARPS = 6;
constexpr int NUM_WARPS = PRODUCER_WARPS + CONSUMER_WARPS;
constexpr int SMEM_STRIDE = TILE_K + WP_SMEM_BANK_PAD;
constexpr int PRODUCER_GROUP_SIZE = PRODUCER_WARPS * 32;
constexpr int TOKENS_PER_WARP = (WP_TOKEN_CHUNK + CONSUMER_WARPS - 1) / CONSUMER_WARPS;

template <int RANK>
__global__ void bgmv_shrink_kernel(const half* __restrict__ X, half* __restrict__ H,
                                    const WarpPipeMetadata* __restrict__ meta, int d_model) {
  constexpr int RANK_GROUPS = (RANK + 31) / 32;  // 1 for RANK<=32, 2 for RANK=64
  constexpr int B_ROW_STRIDE_UNUSED = 0;  // silence unused-constant warnings on some compilers
  (void)B_ROW_STRIDE_UNUSED;

  const int seg_idx = blockIdx.x;
  if (seg_idx >= meta->num_segments) return;
  const SegmentDescriptor seg = meta->segments[seg_idx];

  const int chunk_start = blockIdx.y * WP_TOKEN_CHUNK;
  if (chunk_start >= seg.token_count) return;
  const int chunk_len = min(WP_TOKEN_CHUNK, seg.token_count - chunk_start);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const bool is_producer = wp_is_producer(warp_id, PRODUCER_WARPS);
  const int consumer_warp_id = warp_id - PRODUCER_WARPS;

  extern __shared__ half smem[];
  constexpr int A_BUF_ELEMS = RANK * SMEM_STRIDE;
  constexpr int X_BUF_ELEMS = WP_TOKEN_CHUNK * SMEM_STRIDE;
  constexpr int BUF_ELEMS = A_BUF_ELEMS + X_BUF_ELEMS;
  half* A_buf[2] = {smem, smem + BUF_ELEMS};
  half* X_buf[2] = {smem + A_BUF_ELEMS, smem + BUF_ELEMS + A_BUF_ELEMS};

  const half* A_mat = reinterpret_cast<const half*>(seg.use_staging ? seg.staging_A_ptr : seg.A_ptr);  // [RANK, d_model]
  const half* X_mat = X + (static_cast<int64_t>(seg.token_start) + chunk_start) * d_model;

  const int num_tiles = d_model / TILE_K;

  // acc[slot][g]: this lane's accumulator for owned token `slot` and rank
  // column lane_id + g*32 (only valid while lane_id + g*32 < RANK).
  float acc[TOKENS_PER_WARP][RANK_GROUPS];
#pragma unroll
  for (int i = 0; i < TOKENS_PER_WARP; i++)
#pragma unroll
    for (int g = 0; g < RANK_GROUPS; g++) acc[i][g] = 0.0f;

  auto issue_load = [&](int tile, int stage) {
    if (is_producer) {
      load_tile_async(A_buf[stage], A_mat + tile * TILE_K, RANK, TILE_K, d_model, SMEM_STRIDE, tid,
                       PRODUCER_GROUP_SIZE);
      load_tile_async(X_buf[stage], X_mat + tile * TILE_K, chunk_len, TILE_K, d_model, SMEM_STRIDE, tid,
                       PRODUCER_GROUP_SIZE);
      cp_async_commit();
    }
  };

  issue_load(0, 0);
  cp_async_wait_all();
  wp_barrier_tile_ready();

  for (int t = 0; t < num_tiles; t++) {
    const int stage = t % 2;
    const int next = t + 1;
    if (next < num_tiles) {
      issue_load(next, next % 2);
    }
    if (!is_producer) {
      int slot = 0;
      for (int tok = consumer_warp_id; tok < chunk_len; tok += CONSUMER_WARPS, slot++) {
#pragma unroll
        for (int g = 0; g < RANK_GROUPS; g++) {
          const int r = lane_id + g * 32;
          if (r < RANK) {
            acc[slot][g] += lane_dot_fp16(&A_buf[stage][r * SMEM_STRIDE], &X_buf[stage][tok * SMEM_STRIDE], TILE_K);
          }
        }
      }
    }
    wp_barrier_tile_consumed();
    if (next < num_tiles) {
      cp_async_wait_all();
      wp_barrier_tile_ready();
    }
  }

  if (!is_producer) {
    int slot = 0;
    for (int tok = consumer_warp_id; tok < chunk_len; tok += CONSUMER_WARPS, slot++) {
      const int64_t global_t = static_cast<int64_t>(seg.token_start) + chunk_start + tok;
#pragma unroll
      for (int g = 0; g < RANK_GROUPS; g++) {
        const int r = lane_id + g * 32;
        if (r < RANK) {
          H[global_t * RANK + r] = __float2half(acc[slot][g] * seg.alpha_scale);
        }
      }
    }
  }
}

template <int RANK>
__global__ void bgmv_expand_kernel(const half* __restrict__ H, half* __restrict__ Y,
                                    const WarpPipeMetadata* __restrict__ meta, int out_features) {
  constexpr int B_ROW_STRIDE = RANK + WP_SMEM_BANK_PAD;

  const int seg_idx = blockIdx.x;
  if (seg_idx >= meta->num_segments) return;
  const SegmentDescriptor seg = meta->segments[seg_idx];

  const int chunk_start = blockIdx.y * WP_TOKEN_CHUNK;
  if (chunk_start >= seg.token_count) return;
  const int chunk_len = min(WP_TOKEN_CHUNK, seg.token_count - chunk_start);

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const bool is_producer = wp_is_producer(warp_id, PRODUCER_WARPS);
  const int consumer_warp_id = warp_id - PRODUCER_WARPS;

  extern __shared__ half smem[];
  half* smem_B_buf[2] = {smem, smem + TILE_O * B_ROW_STRIDE};

  const half* B_mat = reinterpret_cast<const half*>(seg.use_staging ? seg.staging_B_ptr : seg.B_ptr);  // [out_features, RANK]
  const int num_o_tiles = out_features / TILE_O;

  for (int ot = 0; ot < num_o_tiles; ot++) {
    const int stage = ot % 2;
    if (is_producer) {
      load_tile_async(smem_B_buf[stage], B_mat + static_cast<int64_t>(ot) * TILE_O * RANK, TILE_O, RANK, RANK,
                       B_ROW_STRIDE, tid, PRODUCER_GROUP_SIZE);
      cp_async_commit();
    }
    cp_async_wait_all();
    wp_barrier_tile_ready();

    if (!is_producer) {
      for (int o_local = consumer_warp_id * 32 + lane_id; o_local < TILE_O; o_local += CONSUMER_WARPS * 32) {
        const int64_t global_o = static_cast<int64_t>(ot) * TILE_O + o_local;
        const half* b_row = &smem_B_buf[stage][o_local * B_ROW_STRIDE];
        for (int tok = 0; tok < chunk_len; tok++) {
          const int64_t global_t = static_cast<int64_t>(seg.token_start) + chunk_start + tok;
          const half* h_row = &H[global_t * RANK];
          // No second alpha_scale multiply here -- same fix as warp_pipe_r32.cu's
          // bgmv_expand_kernel (shrink already applied it once to H).
          const float acc = lane_dot_fp16(b_row, h_row, RANK);
          half* y_ptr = &Y[global_t * out_features + global_o];
          *y_ptr = __float2half(__half2float(*y_ptr) + acc);
        }
      }
    }
    wp_barrier_tile_consumed();
  }
}

template <int RANK>
void launch_bgmv_shrink(const half* X, half* H, const WarpPipeMetadata* meta, int d_model, cudaStream_t stream) {
  dim3 grid(WP_MAX_SEGMENTS, WP_MAX_CHUNKS_PER_SEGMENT, 1);
  dim3 block(NUM_WARPS * 32);
  const size_t smem_bytes = 2 * (RANK * SMEM_STRIDE + WP_TOKEN_CHUNK * SMEM_STRIDE) * sizeof(half);
  // RANK=64's shrink smem (~70KB) exceeds the 48KB default static/dynamic
  // shared memory cap on sm_86 -- a plain launch with smem_bytes > 49152
  // fails at launch time with "invalid argument", not a compile error,
  // since the cap is a runtime opt-in, not a hard architectural limit (Ampere
  // allows up to ~99KB/block dynamic smem with this explicit attribute set).
  // RANK=8/16/32 stay under 48KB and skip this branch entirely. Guarded by a
  // static bool so the opt-in (a real driver-API round trip) happens at most
  // once per process per RANK, not on every launch.
  if (smem_bytes > 48 * 1024) {
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(bgmv_shrink_kernel<RANK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(smem_bytes));
      configured = true;
    }
  }
  bgmv_shrink_kernel<RANK><<<grid, block, smem_bytes, stream>>>(X, H, meta, d_model);
}

template <int RANK>
void launch_bgmv_expand(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features,
                         cudaStream_t stream) {
  constexpr int B_ROW_STRIDE = RANK + WP_SMEM_BANK_PAD;
  dim3 grid(WP_MAX_SEGMENTS, WP_MAX_CHUNKS_PER_SEGMENT, 1);
  dim3 block(NUM_WARPS * 32);
  const size_t smem_bytes = 2 * TILE_O * B_ROW_STRIDE * sizeof(half);
  if (smem_bytes > 48 * 1024) {
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(bgmv_expand_kernel<RANK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(smem_bytes));
      configured = true;
    }
  }
  bgmv_expand_kernel<RANK><<<grid, block, smem_bytes, stream>>>(H, Y, meta, out_features);
}

}  // namespace wp_rgen
