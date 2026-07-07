#pragma once

#include "warp_pipe_metadata.h"

// Host-side owner of the pinned, UVA-mapped SchedulerKernelBridge. Allocated
// once at engine construction; the same device pointer is passed to every
// kernel launch for the life of the process (cudaHostAllocMapped gives a
// fixed address pair -- host pointer for CPU writes, device pointer for GPU
// reads -- with no explicit cudaMemcpy needed).
class SchedulerKernelBridgeHost {
 public:
  SchedulerKernelBridgeHost();
  ~SchedulerKernelBridgeHost();

  SchedulerKernelBridgeHost(const SchedulerKernelBridgeHost&) = delete;
  SchedulerKernelBridgeHost& operator=(const SchedulerKernelBridgeHost&) = delete;

  void write(const float* whittle_scores, const float* t_remaining_ms, const float* gwar_pred_next3,
             const float* lambda_hat, const uint8_t* burst_active, const uint8_t* promo_eligible,
             const uint8_t* is_hot, const uint8_t* tile_size_code, uint32_t num_active_adapters,
             uint32_t num_segments, uint32_t step_id, float global_load);

  const SchedulerKernelBridge* device_ptr() const { return device_ptr_; }
  // Debug-only readback: host_ptr_ is pinned, CPU-writable/readable host
  // memory (cudaHostAllocMapped) -- no GPU read needed. Added so
  // tests/unit/test_metadata_bridge.py can verify write()'s round-trip
  // without needing a throwaway kernel just to echo values back.
  const SchedulerKernelBridge* host_ptr_debug() const { return host_ptr_; }

 private:
  SchedulerKernelBridge* host_ptr_;
  SchedulerKernelBridge* device_ptr_;
};
