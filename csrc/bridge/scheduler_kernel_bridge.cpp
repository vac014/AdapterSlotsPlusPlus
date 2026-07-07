#include "scheduler_kernel_bridge.h"

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

SchedulerKernelBridgeHost::SchedulerKernelBridgeHost() {
  cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&host_ptr_), sizeof(SchedulerKernelBridge),
                                   cudaHostAllocMapped);
  if (err != cudaSuccess) {
    throw std::runtime_error("SchedulerKernelBridgeHost: cudaHostAlloc failed");
  }
  std::memset(host_ptr_, 0, sizeof(SchedulerKernelBridge));
  err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_ptr_), host_ptr_, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_ptr_);
    throw std::runtime_error("SchedulerKernelBridgeHost: cudaHostGetDevicePointer failed");
  }
}

SchedulerKernelBridgeHost::~SchedulerKernelBridgeHost() { cudaFreeHost(host_ptr_); }

void SchedulerKernelBridgeHost::write(const float* whittle_scores, const float* t_remaining_ms,
                                       const float* gwar_pred_next3, const float* lambda_hat,
                                       const uint8_t* burst_active, const uint8_t* promo_eligible,
                                       const uint8_t* is_hot, const uint8_t* tile_size_code,
                                       uint32_t num_active_adapters, uint32_t num_segments, uint32_t step_id,
                                       float global_load) {
  const size_t n = num_active_adapters;
  std::memcpy(host_ptr_->whittle_scores, whittle_scores, n * sizeof(float));
  std::memcpy(host_ptr_->t_remaining_ms, t_remaining_ms, n * sizeof(float));
  std::memcpy(host_ptr_->gwar_pred_next3, gwar_pred_next3, n * sizeof(float));
  std::memcpy(host_ptr_->lambda_hat, lambda_hat, n * sizeof(float));
  std::memcpy(host_ptr_->burst_active, burst_active, n * sizeof(uint8_t));
  std::memcpy(host_ptr_->promo_eligible, promo_eligible, n * sizeof(uint8_t));
  std::memcpy(host_ptr_->is_hot, is_hot, n * sizeof(uint8_t));
  std::memcpy(host_ptr_->tile_size_code, tile_size_code, n * sizeof(uint8_t));
  host_ptr_->num_active_adapters = num_active_adapters;
  host_ptr_->num_segments = num_segments;
  host_ptr_->step_id = step_id;
  host_ptr_->global_load = global_load;
}
