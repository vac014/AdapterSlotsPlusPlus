#pragma once

#include <cstdint>
#include <cuda_runtime.h>  // dim3/cudaStream_t/cudaEvent_t needed by plain .cpp TUs too

// ---------------------------------------------------------------------------
// Compile-time limits.
// ---------------------------------------------------------------------------
static constexpr int WP_MAX_ADAPTERS  = 64;
static constexpr int WP_MAX_SEGMENTS  = 256;
static constexpr int WP_MAX_STAGING   = 8;
static constexpr int WP_TILE_K        = 128;
static constexpr int WP_SMEM_BANK_PAD = 8;
static constexpr int WP_MIN_SEGMENT_TOKENS = 4;
// Largest contiguous same-adapter run any rank-32 kernel call can cover in
// one segment (= WP_TOKEN_CHUNK * WP_MAX_CHUNKS_PER_SEGMENT, the fixed grid.y
// the r32 kernels launch -- see warp_pipe_common.cuh). Host-side callers that
// build segments from real data (e.g. add_lora_packed in
// warp_pipe_bindings.cpp) MUST reject runs longer than this rather than
// silently truncate them -- the kernel's grid simply doesn't have chunk
// slots beyond this length, so any token past it would never be visited.
static constexpr int WP_MAX_SEGMENT_TOKENS = 32;

// Depth split: a segment at or under this many tokens would skip the cp.async
// double-buffer pipeline and run the warp-per-token shuffle path instead
// (wps_shrink_warp/wps_expand_warp in warp_pipe_r32.cu). It is 0, so that path is
// unreachable: the multi-warp pipeline is faster even at the smallest segments, and
// the warp-per-token path is instruction-count-bound rather than memory-bound at this
// size. A_mat/B_mat for one layer+adapter is ~320KB and stays L2-resident across decode
// steps that reuse the same adapters, so its scattered reads cost little and its extra
// per-iteration shuffles buy nothing back. Raising this threshold needs a different
// small-segment algorithm, one that spreads a token's work across more than one warp,
// rather than another access-pattern tweak.
static constexpr int WP_SMALL_SEG_THRESH = 0;

// ---------------------------------------------------------------------------
// SegmentDescriptor: one contiguous same-adapter run of rows in X/Y.
// The static_assert below pins the size: segment_builder.cpp memcpys these at a fixed
// width, so a layout change that slipped past it would corrupt every segment but the first.
// ---------------------------------------------------------------------------
struct alignas(64) SegmentDescriptor {
  int32_t  adapter_id;
  int32_t  token_start;
  int32_t  token_count;
  int32_t  rank;
  int32_t  d_model;
  int32_t  out_features;
  float    alpha_scale;
  uint8_t  promo_eligible;
  uint8_t  burst_active;
  uint8_t  prefetch_ready;
  uint8_t  use_staging;
  uintptr_t A_ptr;
  uintptr_t B_ptr;
  uintptr_t staging_A_ptr;
  uintptr_t staging_B_ptr;
};
// Verified against the real compiler (g++ 13, -std=c++17), not assumed.
static_assert(sizeof(SegmentDescriptor) == 64,
              "SegmentDescriptor size changed -- update segment_builder.cpp memcpy widths");

// ---------------------------------------------------------------------------
// AdapterRun: per-adapter runtime state, written by scheduler_bridge each tick.
// ---------------------------------------------------------------------------
// Spec doc claims 40B; real compiler gives 36B (verified) -- the doc's figure
// was wrong, same class of error as SegmentDescriptor's bridge-version mismatch
// in the prior kernel attempt. Trust the compiler, not the spec, for sizes.
struct AdapterRun {
  int32_t  adapter_id;
  float    whittle_score;
  float    lambda_hat;
  float    gwar_ewma;
  float    gwar_pred_next3;
  float    t_remaining_ms;
  uint32_t queue_depth;
  uint32_t burst_epoch;
  uint8_t  in_mwc;
  uint8_t  is_hot;
  uint8_t  tile_code;
  uint8_t  _pad;
};
static_assert(sizeof(AdapterRun) == 36,
              "AdapterRun size changed -- check scheduler_bridge serialization");

// ---------------------------------------------------------------------------
// SchedulerKernelBridge: pinned, UVA-mapped scheduler<->kernel shared state.
// Written by CPU each scheduler tick; read by GPU kernels via UVA pointer.
// ---------------------------------------------------------------------------
struct alignas(64) SchedulerKernelBridge {
  float    whittle_scores[WP_MAX_ADAPTERS];
  float    t_remaining_ms[WP_MAX_ADAPTERS];
  float    gwar_pred_next3[WP_MAX_ADAPTERS];
  float    lambda_hat[WP_MAX_ADAPTERS];
  uint8_t  burst_active[WP_MAX_ADAPTERS];
  uint8_t  promo_eligible[WP_MAX_ADAPTERS];
  uint8_t  is_hot[WP_MAX_ADAPTERS];
  uint8_t  prefetch_ready[WP_MAX_ADAPTERS];
  uint8_t  tile_size_code[WP_MAX_ADAPTERS];
  uint32_t step_id;
  uint32_t num_active_adapters;
  uint32_t num_segments;
  float    global_load;
};
static_assert(sizeof(SchedulerKernelBridge) <= 2048, "SKB must fit in 2 KB");

// ---------------------------------------------------------------------------
// PrefetchDescriptor: one per adapter queued for prefetch into staging.
// Not exercised here (prefetch_kernel.cu is a stub) -- the struct is kept so the
// metadata layout stays fixed whether or not the prefetch path is compiled in.
// ---------------------------------------------------------------------------
struct PrefetchDescriptor {
  int32_t   adapter_id;
  uint32_t  staging_slot;
  uintptr_t A_src;
  uintptr_t B_src;
  uintptr_t A_dst;
  uintptr_t B_dst;
  uint32_t  A_bytes;
  uint32_t  B_bytes;
  float     whittle_priority;
  uintptr_t ready_flag;  // device pointer to a cuda::atomic<uint32_t>
};
static_assert(sizeof(PrefetchDescriptor) == 64, "PrefetchDescriptor size changed");

// ---------------------------------------------------------------------------
// WarpPipeMetadata: the graph-safety-critical struct passed to every kernel
// launch. Grid/block dims and which kernel is launched are CONSTANT across
// calls (see warp_pipe_r32.cu's launch wrappers); only segments[]/num_segments
// vary, and the kernel reads num_segments at runtime, never as a host scalar.
// ---------------------------------------------------------------------------
struct WarpPipeMetadata {
  SegmentDescriptor segments[WP_MAX_SEGMENTS];
  uint32_t          num_segments;
  const SchedulerKernelBridge* bridge_ptr;
};

struct KernelLaunchDescriptor {
  int      rank;
  int      tile_k;
  int      num_warps;
  int      producer_warps;
  size_t   smem_bytes;
  dim3     grid;
  dim3     block;
  cudaStream_t stream;
  bool     use_staging;
};

struct GWARMetadata {
  float   gwar[WP_MAX_ADAPTERS];
  float   gwar_ewma[WP_MAX_ADAPTERS];
  float   gwar_pred_3step[WP_MAX_ADAPTERS];
  uint8_t pipeline_eligible[WP_MAX_ADAPTERS];
};

struct ErlangMetadata {
  float    lambda_hat[WP_MAX_ADAPTERS];
  float    t_fill_90pct[WP_MAX_ADAPTERS];
  float    t_remaining_ms[WP_MAX_ADAPTERS];
  uint32_t queue_depth[WP_MAX_ADAPTERS];
  uint32_t W;
};

struct AdapterPopularityState {
  float    zipf_alpha;
  uint32_t hot_ids[WP_MAX_ADAPTERS];
  float    hot_scores[WP_MAX_ADAPTERS];
  uint32_t num_hot;
  uint32_t k_hot;
};

struct RuntimeExecutionState {
  uint32_t           step_id;
  uint32_t           num_segments;
  uint32_t           total_tokens;
  SegmentDescriptor* segs_dev;
  WarpPipeMetadata*  meta_dev;
  SchedulerKernelBridge* bridge_ptr;
  GWARMetadata       gwar;
  ErlangMetadata     erlang;
  AdapterPopularityState popularity;
  cudaStream_t       stream_main;
  cudaStream_t       stream_prefetch;
  cudaEvent_t        prefetch_done;
  cudaEvent_t        step_done;
};
