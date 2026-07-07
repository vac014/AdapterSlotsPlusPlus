// warp_pipe_r32.cu -- rank=32 shrink+expand.
//
// Two departures from the obvious design, both of them measured rather than
// assumed:
//
//   1. GRID SHAPE: the natural reading of a "persistent segment loop" is
//      grid=(1,1,1), one block walking every segment serially. Benchmarked
//      against vLLM's stock bgmv_shrink/bgmv_expand, that is 3-150x SLOWER,
//      because it occupies exactly 1 of the GPU's 84 SMs while the stock
//      kernel spreads across all of them. Instead: one block per (segment,
//      token-chunk) pair, grid.x = WP_MAX_SEGMENTS, grid.y =
//      WP_MAX_CHUNKS_PER_SEGMENT, constant shape every launch (so it stays
//      CUDA-graph safe), and out-of-range blocks exit immediately.
//
//   2. OUTPUT WRITE PATTERN: RANK=32=warpSize is exploited in BOTH kernels.
//      Every lane owns one full output element and carries its own reduction
//      in registers, so there is no __shfl reduction and no lane-0-only
//      scalar write: all 32 lanes write together, every time.
//
// cp.async granularity and the compute primitives have their own
// non-obvious constraints; see cp_async_utils.cuh and mma_utils.cuh.

#include "warp_pipe_r32.h"
#include "warp_pipe_common.cuh"
#include "cp_async_utils.cuh"
#include "mma_utils.cuh"
#include "pipeline_barrier.cuh"

namespace wp_r32 {

// TILE_K doubled from spec's WP_TILE_K=128 to halve the per-block tile loop
// (32->16 iterations), directly targeting the 33-35% barrier-stall finding
// at K=16,T=16 (each tile costs 2 __syncthreads() round-trips). Two prior
// occupancy-based fixes for that same slowdown were tried and reverted after
// measuring worse despite improving their target metric -- see
// mma_utils.cuh and warp_pipe_common.cuh's CONSUMER_WARPS comment.
constexpr int TILE_K = 256;
constexpr int RANK = 32;
constexpr int PRODUCER_WARPS = WP_RANK32_PRODUCER_WARPS;     // 2
constexpr int CONSUMER_WARPS = WP_RANK32_CONSUMER_WARPS;     // 6
constexpr int NUM_WARPS = WP_RANK32_NUM_WARPS;                // 8
constexpr int SMEM_STRIDE = TILE_K + WP_SMEM_BANK_PAD;        // 136
constexpr int PRODUCER_GROUP_SIZE = PRODUCER_WARPS * 32;      // 64
constexpr int TOKENS_PER_WARP = (WP_TOKEN_CHUNK + CONSUMER_WARPS - 1) / CONSUMER_WARPS;  // 2

// Doubled alongside TILE_K, same rationale: halves expand's tile loop
// (32->16 iterations) to cut barrier-sync round-trips at large K,T.
constexpr int TILE_O = 256;
constexpr int B_ROW_STRIDE = RANK + WP_SMEM_BANK_PAD;  // 40

// ── Shrink ──────────────────────────────────────────────────────────────────
// One block per (segment, token-chunk). Each of the 6 consumer warps owns a
// strided subset of this chunk's tokens; within a warp, lane r (=lane_id)
// owns rank-output r and computes its own dot product over the d_model
// reduction, tiled via the cp.async double-buffer pipeline.
__global__ void bgmv_shrink_kernel(const half* __restrict__ X,
                                    half* __restrict__ H,
                                    const WarpPipeMetadata* __restrict__ meta,
                                    int d_model) {
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

  const int num_tiles = d_model / TILE_K;  // assumes d_model % TILE_K == 0 (true: 4096/128=32)

  // Each owned token gets its own accumulator slot; lane r's value within
  // that slot IS the rank-r output for that token (no cross-lane combine).
  float acc[TOKENS_PER_WARP];
#pragma unroll
  for (int i = 0; i < TOKENS_PER_WARP; i++) acc[i] = 0.0f;

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
        acc[slot] += lane_dot_fp16(&A_buf[stage][lane_id * SMEM_STRIDE], &X_buf[stage][tok * SMEM_STRIDE], TILE_K);
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
      H[global_t * RANK + lane_id] = __float2half(acc[slot] * seg.alpha_scale);
    }
  }
}

// ── Expand ──────────────────────────────────────────────────────────────────
// One block per (segment, token-chunk), same grid as shrink. Loops over
// out_features in TILE_O-wide tiles; within each tile, every lane (across all
// consumer warps) owns one output column and writes it directly. That output
// ownership is what keeps expand cheap: writing it lane-0-scalar instead costs
// roughly 7x, and makes expand the dominant kernel rather than the cheaper of
// the two.
__global__ void bgmv_expand_kernel(const half* __restrict__ H,
                                    half* __restrict__ Y,
                                    const WarpPipeMetadata* __restrict__ meta,
                                    int out_features) {
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
  const int num_o_tiles = out_features / TILE_O;  // assumes out_features % TILE_O == 0

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
          // FIX: do NOT multiply by seg.alpha_scale here. Real LoRA/vLLM
          // semantics apply the alpha/rank scale exactly once -- confirmed
          // against vllm/lora/ops/bgmv_shrink.py (has a `scaling` arg) vs
          // bgmv_expand.py (no scale arg at all, only add_inputs). shrink
          // already applies seg.alpha_scale to H below; applying it again
          // here was squaring it. Caught via add_lora_packed's correctness
          // test cross-checked against vLLM's real add_lora() semantics --
          // every earlier "ALL PASS" in this project validated kernel-
          // matches-spec-doc's-own-reference, not kernel-matches-real-math,
          // since reference_bgmv.py mirrored the same doubled convention.
          const float acc = lane_dot_fp16(b_row, h_row, RANK);
          half* y_ptr = &Y[global_t * out_features + global_o];
          *y_ptr = __float2half(__half2float(*y_ptr) + acc);
        }
      }
    }
    wp_barrier_tile_consumed();
  }
}

// ── warp-per-token small-segment path ──────────────────────────────────────
// For segments at or under WP_SMALL_SEG_THRESH
// tokens, the 256-thread/8-warp cp.async pipeline above pays its fixed setup
// cost (double-buffer priming, multiple __syncthreads rounds across
// NUM_WARPS warps) for too little work to amortize it. Below, ONE warp
// handles ONE entire token end-to-end -- no shared memory, no pipeline --
// exploiting RANK=32=warpSize so a single coalesced 32-wide load plus
// __shfl_sync broadcast stands in for what the pipeline path does via shared
// memory tiling.
//
// wps_expand_warp specifically targets a real inefficiency in
// bgmv_expand_kernel above: every lane there reads the full RANK=32-element
// H row from global memory independently inside lane_dot_fp16. The redundant
// H reads across lanes are the real cost here (it is not a lane-0-only
// bottleneck). So each lane loads exactly one H element, once, and shares it
// with the other 31 by shuffle.
__device__ __forceinline__ void wps_shrink_warp(const half* __restrict__ x_row,
                                                 const half* __restrict__ A_mat, int d_model,
                                                 float alpha_scale, half* __restrict__ h_row) {
  const int lane = threadIdx.x & 31;
  float acc = 0.0f;
  for (int base = 0; base < d_model; base += 32) {
    const float xv = __half2float(x_row[base + lane]);
#pragma unroll
    for (int k = 0; k < 32; k++) {
      const float xb = __shfl_sync(0xffffffffu, xv, k);
      acc += xb * __half2float(A_mat[lane * d_model + base + k]);
    }
  }
  h_row[lane] = __float2half(acc * alpha_scale);
}

__device__ __forceinline__ void wps_expand_warp(const half* __restrict__ h_row,
                                                 const half* __restrict__ B_mat, int out_features,
                                                 half* __restrict__ y_row) {
  const int lane = threadIdx.x & 31;
  const float h_lane = __half2float(h_row[lane]);
  for (int o = lane; o < out_features; o += 32) {
    const half* b_row = &B_mat[static_cast<int64_t>(o) * RANK];
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < 32; r++) {
      const float hb = __shfl_sync(0xffffffffu, h_lane, r);
      acc += __half2float(b_row[r]) * hb;
    }
    y_row[o] = __float2half(__half2float(y_row[o]) + acc);
  }
}

// ── DSAK live-path kernels ──────────────────────────────────────────────────
// Segment-BOUNDARY construction (adapter_id/token_start/token_count) happens
// outside these kernels and outside add_lora_packed() entirely --
// once per decode STEP, in build_seg_bounds_kernel below,
// triggered from PunicaWrapper.update_metadata() (always-uncaptured,
// verified against vllm/worker/model_runner.py's real call chain), instead
// of once per LAYER CALL the way build_segments_kernel above did. A 13B
// model decode step calls add_lora() ~80 times (one per LoRA-enabled linear
// layer); the segment boundaries are identical every one of those calls
// within a step -- only A_ptr/B_ptr (this layer's wa_t_all/wb_t_all) differ
// -- so rebuilding them 80x was pure waste.
//
// Consequence: these kernels do NOT trust seg.A_ptr/seg.B_ptr/seg.rank/
// seg.d_model/seg.out_features/seg.alpha_scale/seg.use_staging (those fields
// are left stale/zero by build_seg_bounds_kernel) -- they take wa_base/
// wb_base/strides/scale/d_model/out_features as direct kernel arguments
// instead, computing this call's A_ptr/B_ptr from wa_base/wb_base +
// seg.adapter_id, the same pattern build_segments_kernel already used for
// pointer freshness. Only seg.adapter_id/token_start/token_count are read.
//
// Deliberately duplicated rather than sharing bgmv_shrink_kernel/
// bgmv_expand_kernel's body: those two remain struct-A_ptr/B_ptr-driven for
// shrink()/expand()/build_segments() (the scheduler-bridge/benchmark path --
// bench_vs_punica.py, test_bgmv_correctness.py, profile_breakdown.py,
// test_scheduler_bridge.py all call through that path and must keep working
// unmodified). Splitting cp.async pipeline logic into a __device__ function
// parameterized over both call conventions is possible, but it would put that
// path at risk for no measured benefit -- not
// worth it given this is the live serving path's first rework.
__global__ void bgmv_shrink_dsak_kernel(const half* __restrict__ X, half* __restrict__ H,
                                         const WarpPipeMetadata* __restrict__ meta,
                                         const half* __restrict__ wa_base, int64_t wa_stride0,
                                         float alpha_scale, int d_model) {
  const int seg_idx = blockIdx.x;
  if (seg_idx >= meta->num_segments) return;
  const SegmentDescriptor seg = meta->segments[seg_idx];

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const half* A_mat = wa_base + static_cast<int64_t>(seg.adapter_id) * wa_stride0;

  if (seg.token_count <= WP_SMALL_SEG_THRESH) {
    if (blockIdx.y != 0) return;
    if (warp_id >= seg.token_count) return;
    const int64_t global_t = static_cast<int64_t>(seg.token_start) + warp_id;
    wps_shrink_warp(X + global_t * d_model, A_mat, d_model, alpha_scale, H + global_t * RANK);
    return;
  }

  const int chunk_start = blockIdx.y * WP_TOKEN_CHUNK;
  if (chunk_start >= seg.token_count) return;
  const int chunk_len = min(WP_TOKEN_CHUNK, seg.token_count - chunk_start);

  const bool is_producer = wp_is_producer(warp_id, PRODUCER_WARPS);
  const int consumer_warp_id = warp_id - PRODUCER_WARPS;

  extern __shared__ half smem[];
  constexpr int A_BUF_ELEMS = RANK * SMEM_STRIDE;
  constexpr int X_BUF_ELEMS = WP_TOKEN_CHUNK * SMEM_STRIDE;
  constexpr int BUF_ELEMS = A_BUF_ELEMS + X_BUF_ELEMS;
  half* A_buf[2] = {smem, smem + BUF_ELEMS};
  half* X_buf[2] = {smem + A_BUF_ELEMS, smem + BUF_ELEMS + A_BUF_ELEMS};

  const half* X_mat = X + (static_cast<int64_t>(seg.token_start) + chunk_start) * d_model;
  const int num_tiles = d_model / TILE_K;

  float acc[TOKENS_PER_WARP];
#pragma unroll
  for (int i = 0; i < TOKENS_PER_WARP; i++) acc[i] = 0.0f;

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
        acc[slot] += lane_dot_fp16(&A_buf[stage][lane_id * SMEM_STRIDE], &X_buf[stage][tok * SMEM_STRIDE], TILE_K);
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
      H[global_t * RANK + lane_id] = __float2half(acc[slot] * alpha_scale);
    }
  }
}

__global__ void bgmv_expand_dsak_kernel(const half* __restrict__ H, half* __restrict__ Y,
                                         const WarpPipeMetadata* __restrict__ meta,
                                         const half* __restrict__ wb_base, int64_t wb_stride0,
                                         int out_features) {
  const int seg_idx = blockIdx.x;
  if (seg_idx >= meta->num_segments) return;
  const SegmentDescriptor seg = meta->segments[seg_idx];

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;

  const half* B_mat = wb_base + static_cast<int64_t>(seg.adapter_id) * wb_stride0;

  if (seg.token_count <= WP_SMALL_SEG_THRESH) {
    if (blockIdx.y != 0) return;
    if (warp_id >= seg.token_count) return;
    const int64_t global_t = static_cast<int64_t>(seg.token_start) + warp_id;
    wps_expand_warp(H + global_t * RANK, B_mat, out_features, Y + global_t * out_features);
    return;
  }

  const int chunk_start = blockIdx.y * WP_TOKEN_CHUNK;
  if (chunk_start >= seg.token_count) return;
  const int chunk_len = min(WP_TOKEN_CHUNK, seg.token_count - chunk_start);

  const bool is_producer = wp_is_producer(warp_id, PRODUCER_WARPS);
  const int consumer_warp_id = warp_id - PRODUCER_WARPS;

  extern __shared__ half smem[];
  half* smem_B_buf[2] = {smem, smem + TILE_O * B_ROW_STRIDE};
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
          const float acc = lane_dot_fp16(b_row, h_row, RANK);
          half* y_ptr = &Y[global_t * out_features + global_o];
          *y_ptr = __float2half(__half2float(*y_ptr) + acc);
        }
      }
    }
    wp_barrier_tile_consumed();
  }
}

// ── once-per-step segment boundary builder ─────────────────────────────────
// Fixed-grid, no-host-sync, writes ONLY
// adapter_id/token_start/token_count (no A_ptr/B_ptr/rank/d_model/
// out_features/alpha_scale -- bgmv_shrink_dsak_kernel/bgmv_expand_dsak_kernel
// take those as direct per-call kernel arguments instead, see above). Called
// once per decode step from FusedPunicaWrapperWarpPipe.update_metadata()
// (punica_wrapper.py), not once per layer: this boundary structure is identical
// across all ~80 add_lora() calls in a step, so building it per layer would be
// redundant 79 times out of 80.
//
// NOT single-threaded (unlike build_segments_kernel above, whose serial scan
// over total_tokens is fine there since it was always bounded by one call's
// own segment count, never re-measured at scale): one thread per token
// position (block size = WP_MAX_SEGMENTS = the upper bound on total_tokens,
// enforced by build_seg_bounds()'s caller-side check). Only RUN-START
// threads (token_lora_indices[i] != token_lora_indices[i-1]) do any work,
// and each one scans forward only across ITS OWN run -- bounded by that
// run's length, not total_tokens. Different runs' start threads execute
// concurrently across warps, so wall-clock cost scales with the LONGEST
// single run, not with K (the number of distinct adapters / runs in the
// batch). A naive single-thread version (the original build_seg_bounds_kernel
// here) would instead serialize through every token across every run, making
// wall-clock cost grow with K even when each adapter only contributes a
// handful of tokens -- exactly the O(K) regression to avoid, since K is the
// dimension this project's K-sweep benchmarks (K=2/4/8/...) scale up.
// Each run-start thread's output slot range is allocated via a block-wide
// parallel prefix sum (see kernel body) rather than atomicAdd -- an
// atomicAdd-based version of this was tried and measured (bench_seg_bounds_
// scaling.py) to still scale with K, since every concurrent run's atomicAdd
// hits the same address and atomics to one address serialize in hardware.
// The prefix sum makes slot order match TOKEN order (run-start thread i's
// slot range always precedes thread j>i's), a deterministic side effect of
// the scan, not something correctness depends on -- downstream consumers
// (bgmv_shrink_dsak_kernel/bgmv_expand_dsak_kernel) process segments
// independently and write to disjoint token ranges, so order never affects
// correctness.
__global__ void build_seg_bounds_kernel(const int64_t* __restrict__ token_lora_indices, int total_tokens,
                                         WarpPipeMetadata* __restrict__ meta) {
  if (blockIdx.x != 0) return;
  const int i = threadIdx.x;

  // Slot allocation is a block-wide O(log WP_MAX_SEGMENTS) inclusive prefix sum
  // (Hillis-Steele) over each thread's segment-count contribution, which hands every
  // run-start thread its own output slot range. An atomicAdd on &meta->num_segments
  // would be simpler and does not work here: every run-start thread's atomic lands on
  // the SAME address, atomics to one address serialize in hardware, and the cost would
  // therefore grow with the NUMBER of concurrent runs (K) even though each run's own
  // scan is already parallel.
  __shared__ int contrib[WP_MAX_SEGMENTS];

  // run_len only needs to be visible to the SAME thread that computed it
  // (each run-start thread writes its own segments below using its own
  // run_len register) -- no other thread ever needs it, so it stays a
  // plain register, not a second shared array.
  int64_t my_idx = -1;
  int run_len = 0;
  int my_contrib = 0;
  if (i < total_tokens) {
    my_idx = token_lora_indices[i];
    const bool is_run_start = (i == 0) || (token_lora_indices[i - 1] != my_idx);
    if (is_run_start && my_idx >= 0) {
      run_len = 1;
      while (i + run_len < total_tokens && token_lora_indices[i + run_len] == my_idx) run_len++;
      my_contrib = (run_len + WP_MAX_SEGMENT_TOKENS - 1) / WP_MAX_SEGMENT_TOKENS;
    }
  }
  contrib[i] = my_contrib;
  __syncthreads();

  for (int offset = 1; offset < WP_MAX_SEGMENTS; offset <<= 1) {
    const int v = (i >= offset) ? contrib[i - offset] : 0;
    __syncthreads();
    contrib[i] += v;
    __syncthreads();
  }
  // contrib[i] is now an INCLUSIVE prefix sum; this thread's own slot range
  // starts right after everyone before it, i.e. at the EXCLUSIVE sum.
  const int base_slot = contrib[i] - my_contrib;
  if (i == WP_MAX_SEGMENTS - 1) meta->num_segments = static_cast<uint32_t>(contrib[i]);

  if (my_contrib == 0) return;
  for (int s = 0; s < my_contrib && base_slot + s < WP_MAX_SEGMENTS; s++) {
    SegmentDescriptor& sd = meta->segments[base_slot + s];
    sd.adapter_id = static_cast<int32_t>(my_idx);
    sd.token_start = i + s * WP_MAX_SEGMENT_TOKENS;
    sd.token_count = min(WP_MAX_SEGMENT_TOKENS, run_len - s * WP_MAX_SEGMENT_TOKENS);
  }
}

// ── Device-side segment builder ────────────────────────────────────────────
// Replaces add_lora_packed()'s old host-side path: a .cpu() sync on
// token_lora_indices to read its values, then a serial CPU loop building
// SegmentDescriptor entries -- both illegal mid-CUDA-graph-capture (the sync
// errors with "operation not permitted when stream is capturing"; even if it
// didn't, the resulting *launch* of bgmv_shrink/expand wouldn't be the
// problem, since their grid is already fixed -- see this kernel's own header
// comment -- but reaching that launch at all required the now-removed sync).
//
// Single thread, serial scan over total_tokens (<=WP_MAX_SEGMENTS=256 in
// every config this project uses -- vLLM's own max_num_seqs default). Fast
// enough not to matter (microseconds over <=256 int64 reads) and entirely
// on-device: reads token_lora_indices via plain global loads, writes
// SegmentDescriptor entries via plain global stores into meta (the same
// cudaHostAllocMapped memory bgmv_shrink/expand already read from -- no
// extra copy, ordinary same-stream launch-order consistency applies).
//
// Splits any run longer than WP_MAX_SEGMENT_TOKENS into multiple consecutive
// segments (same adapter_id/A_ptr/B_ptr) rather than rejecting it -- the old
// host path returned false (forcing a stock-kernel fallback) for this case;
// splitting is possible here because there is no per-call Python-level
// fallback decision to make anymore (graph capture bakes in one path
// permanently, so "fall back sometimes" was never viable for the captured
// case to begin with). This is provably safe under this project's configs:
// worst case needed segments = sum(ceil(run_len/32)) <= total_tokens <=
// WP_MAX_SEGMENTS, since splitting a run of length L into 32-token pieces
// never produces more than L segments.
__global__ void build_segments_kernel(const int64_t* __restrict__ token_lora_indices,
                                       int total_tokens,
                                       const half* __restrict__ wa_base, int64_t wa_stride0,
                                       const half* __restrict__ wb_base, int64_t wb_stride0,
                                       float alpha_scale, int rank, int d_model, int out_features,
                                       WarpPipeMetadata* __restrict__ meta) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t num_segs = 0;
  int i = 0;
  while (i < total_tokens) {
    const int64_t lora_idx = token_lora_indices[i];
    int j = i + 1;
    while (j < total_tokens && token_lora_indices[j] == lora_idx) j++;
    if (lora_idx >= 0) {
      int seg_start = i;
      while (seg_start < j && num_segs < WP_MAX_SEGMENTS) {
        const int seg_len = min(WP_MAX_SEGMENT_TOKENS, j - seg_start);
        SegmentDescriptor& sd = meta->segments[num_segs];
        sd.adapter_id = static_cast<int32_t>(lora_idx);
        sd.token_start = seg_start;
        sd.token_count = seg_len;
        sd.rank = rank;
        sd.d_model = d_model;
        sd.out_features = out_features;
        sd.alpha_scale = alpha_scale;
        sd.promo_eligible = 0;
        sd.burst_active = 0;
        sd.use_staging = 0;
        sd.prefetch_ready = 0;
        sd.A_ptr = reinterpret_cast<uintptr_t>(wa_base + lora_idx * wa_stride0);
        sd.B_ptr = reinterpret_cast<uintptr_t>(wb_base + lora_idx * wb_stride0);
        sd.staging_A_ptr = 0;
        sd.staging_B_ptr = 0;
        num_segs++;
        seg_start += seg_len;
      }
    }
    i = j;
  }
  meta->num_segments = num_segs;
}

}  // namespace wp_r32

void launch_build_segments_r32(const int64_t* token_lora_indices, int total_tokens,
                                const half* wa_base, int64_t wa_stride0,
                                const half* wb_base, int64_t wb_stride0,
                                float alpha_scale, int rank, int d_model, int out_features,
                                WarpPipeMetadata* meta, cudaStream_t stream) {
  wp_r32::build_segments_kernel<<<1, 1, 0, stream>>>(
      token_lora_indices, total_tokens, wa_base, wa_stride0, wb_base, wb_stride0,
      alpha_scale, rank, d_model, out_features, meta);
}

void launch_bgmv_shrink_r32(const half* X, half* H, const WarpPipeMetadata* meta, int d_model,
                             cudaStream_t stream) {
  dim3 grid(WP_MAX_SEGMENTS, WP_MAX_CHUNKS_PER_SEGMENT, 1);
  dim3 block(wp_r32::NUM_WARPS * 32);
  constexpr size_t smem_bytes =
      2 * (wp_r32::RANK * wp_r32::SMEM_STRIDE + WP_TOKEN_CHUNK * wp_r32::SMEM_STRIDE) * sizeof(half);
  wp_r32::bgmv_shrink_kernel<<<grid, block, smem_bytes, stream>>>(X, H, meta, d_model);
}

void launch_bgmv_expand_r32(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features,
                             cudaStream_t stream) {
  dim3 grid(WP_MAX_SEGMENTS, WP_MAX_CHUNKS_PER_SEGMENT, 1);
  dim3 block(wp_r32::NUM_WARPS * 32);
  constexpr size_t smem_bytes = 2 * wp_r32::TILE_O * wp_r32::B_ROW_STRIDE * sizeof(half);
  wp_r32::bgmv_expand_kernel<<<grid, block, smem_bytes, stream>>>(H, Y, meta, out_features);
}

void launch_build_seg_bounds_r32(const int64_t* token_lora_indices, int total_tokens, WarpPipeMetadata* meta,
                                  cudaStream_t stream) {
  // One thread per possible token position (WP_MAX_SEGMENTS upper-bounds
  // total_tokens -- enforced in build_seg_bounds()/bindings.cpp before this
  // launch), not one thread total -- see build_seg_bounds_kernel's comment
  // for why this is what makes per-step cost independent of K.
  wp_r32::build_seg_bounds_kernel<<<1, WP_MAX_SEGMENTS, 0, stream>>>(token_lora_indices, total_tokens, meta);
}

void launch_bgmv_shrink_dsak_r32(const half* X, half* H, const WarpPipeMetadata* meta, const half* wa_base,
                                  int64_t wa_stride0, float alpha_scale, int d_model, cudaStream_t stream) {
  dim3 grid(WP_MAX_SEGMENTS, WP_MAX_CHUNKS_PER_SEGMENT, 1);
  dim3 block(wp_r32::NUM_WARPS * 32);
  constexpr size_t smem_bytes =
      2 * (wp_r32::RANK * wp_r32::SMEM_STRIDE + WP_TOKEN_CHUNK * wp_r32::SMEM_STRIDE) * sizeof(half);
  wp_r32::bgmv_shrink_dsak_kernel<<<grid, block, smem_bytes, stream>>>(X, H, meta, wa_base, wa_stride0, alpha_scale,
                                                                        d_model);
}

void launch_bgmv_expand_dsak_r32(const half* H, half* Y, const WarpPipeMetadata* meta, const half* wb_base,
                                  int64_t wb_stride0, int out_features, cudaStream_t stream) {
  dim3 grid(WP_MAX_SEGMENTS, WP_MAX_CHUNKS_PER_SEGMENT, 1);
  dim3 block(wp_r32::NUM_WARPS * 32);
  constexpr size_t smem_bytes = 2 * wp_r32::TILE_O * wp_r32::B_ROW_STRIDE * sizeof(half);
  wp_r32::bgmv_expand_dsak_kernel<<<grid, block, smem_bytes, stream>>>(H, Y, meta, wb_base, wb_stride0, out_features);
}
