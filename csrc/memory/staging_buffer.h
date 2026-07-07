#pragma once

#include <cstdint>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../bridge/warp_pipe_metadata.h"

// Early-Copy Prefetch Engine staging pool.
//
// Prefetching adapter weights is only worth its complexity if they are not
// already resident in L2. Measured: bgmv_shrink_kernel's L2 hit rate is
// 61.9-63.5% at K=8/16 and bgmv_expand_kernel's is 41.0% at K=16, so a
// substantial fraction of every adapter read goes to device memory and there
// is real latency here to hide.
//
// Each slot owns a persistent device buffer pair (A_buf, B_buf) sized for
// the largest A/B matrix this kernel build supports (rank=32 -- the only
// rank warp_pipe_r32.cu handles -- times WP_STAGING_MAX_DMODEL). A prefetch
// copy (prefetch_kernel.cu, issued on a dedicated internal stream) writes
// into a slot; a cudaEvent_t recorded right after the copy is what
// is_ready() polls (cudaEventQuery, non-blocking) to know when the segment
// builder may safely point a kernel's use_staging/staging_A_ptr at it.
static constexpr int WP_STAGING_MAX_DMODEL = 8192;  // covers this project's 4096/5120 d_model and o_proj/down_proj out_features
static constexpr int WP_STAGING_RANK = 32;          // this kernel build's only supported rank

class StagingBufferPool {
 public:
  explicit StagingBufferPool(int num_slots)
      : slot_adapter_id_(num_slots, -1), a_buf_(num_slots, nullptr), b_buf_(num_slots, nullptr),
        ready_event_(num_slots, nullptr), copy_issued_(num_slots, false) {
    const size_t a_bytes = static_cast<size_t>(WP_STAGING_RANK) * WP_STAGING_MAX_DMODEL * sizeof(half);
    const size_t b_bytes = static_cast<size_t>(WP_STAGING_MAX_DMODEL) * WP_STAGING_RANK * sizeof(half);
    for (int i = 0; i < num_slots; i++) {
      cudaMalloc(&a_buf_[i], a_bytes);
      cudaMalloc(&b_buf_[i], b_bytes);
      cudaEventCreateWithFlags(&ready_event_[i], cudaEventDisableTiming);
    }
  }

  ~StagingBufferPool() {
    for (size_t i = 0; i < a_buf_.size(); i++) {
      if (a_buf_[i]) cudaFree(a_buf_[i]);
      if (b_buf_[i]) cudaFree(b_buf_[i]);
      if (ready_event_[i]) cudaEventDestroy(ready_event_[i]);
    }
  }

  int find_slot_for_adapter(int32_t adapter_id) const {
    for (size_t i = 0; i < slot_adapter_id_.size(); i++) {
      if (slot_adapter_id_[i] == adapter_id) return static_cast<int>(i);
    }
    return -1;
  }

  // Returns an existing slot for adapter_id if already assigned, else the
  // first free slot, else -1 (caller must skip prefetch this tick -- never
  // a correctness issue, only a missed optimization opportunity, same
  // always-safe-fallback pattern as everything else in this project).
  int acquire_slot(int32_t adapter_id) {
    int existing = find_slot_for_adapter(adapter_id);
    if (existing >= 0) return existing;
    for (size_t i = 0; i < slot_adapter_id_.size(); i++) {
      if (slot_adapter_id_[i] == -1) {
        slot_adapter_id_[i] = adapter_id;
        copy_issued_[i] = false;
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  void release_slot(int slot) {
    if (slot < 0 || slot >= static_cast<int>(slot_adapter_id_.size())) return;
    slot_adapter_id_[slot] = -1;
    copy_issued_[slot] = false;
  }

  void mark_copy_issued(int slot, cudaStream_t prefetch_stream) {
    copy_issued_[slot] = true;
    cudaEventRecord(ready_event_[slot], prefetch_stream);
  }

  // Non-blocking: cudaEventQuery, never cudaEventSynchronize -- this is
  // polled from the same eager Python call path as everything else in this
  // project and must never introduce a host stall.
  bool is_ready(int slot) const {
    if (slot < 0 || slot >= static_cast<int>(slot_adapter_id_.size())) return false;
    if (!copy_issued_[slot]) return false;
    return cudaEventQuery(ready_event_[slot]) == cudaSuccess;
  }

  void* a_ptr(int slot) const { return a_buf_[slot]; }
  void* b_ptr(int slot) const { return b_buf_[slot]; }
  int32_t adapter_of(int slot) const { return slot_adapter_id_[slot]; }

 private:
  std::vector<int32_t> slot_adapter_id_;
  std::vector<void*> a_buf_;
  std::vector<void*> b_buf_;
  std::vector<cudaEvent_t> ready_event_;
  std::vector<bool> copy_issued_;
};
