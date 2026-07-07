#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <unordered_map>

#include "../bridge/scheduler_kernel_bridge.h"
#include "../bridge/warp_pipe_metadata.h"
#include "../dispatch/segment_builder.h"
#include "../dispatch/warp_pipe_dispatcher.h"
#include "../kernels/prefetch_kernel.h"
#include "../memory/adapter_store.h"
#include "../memory/staging_buffer.h"

namespace py = pybind11;

class WarpPipeExtension {
 public:
  WarpPipeExtension() : staging_(WP_MAX_STAGING), bridge_() {
    cudaError_t err =
        cudaHostAlloc(reinterpret_cast<void**>(&meta_host_), sizeof(WarpPipeMetadata), cudaHostAllocMapped);
    TORCH_CHECK(err == cudaSuccess, "WarpPipeExtension: cudaHostAlloc(WarpPipeMetadata) failed");
    std::memset(meta_host_, 0, sizeof(WarpPipeMetadata));
    err = cudaHostGetDevicePointer(reinterpret_cast<void**>(&meta_dev_), meta_host_, 0);
    TORCH_CHECK(err == cudaSuccess, "WarpPipeExtension: cudaHostGetDevicePointer failed");
    meta_host_->bridge_ptr = bridge_.device_ptr();
    // the prefetch engine's own stream -- deliberately NOT torch's current stream:
    // the entire point is for the prefetch copy to run concurrently with
    // whatever stream_main work (the shrink/expand kernels) is doing, not
    // serialize behind it. Never touched by anything but prefetch_adapter().
    err = cudaStreamCreateWithFlags(&stream_prefetch_, cudaStreamNonBlocking);
    TORCH_CHECK(err == cudaSuccess, "WarpPipeExtension: cudaStreamCreateWithFlags failed");
  }

  ~WarpPipeExtension() {
    cudaFreeHost(meta_host_);
    cudaStreamDestroy(stream_prefetch_);
  }

  // Issue an async device-to-device copy of adapter_id's A/B weights into a
  // staging slot on stream_prefetch_, concurrent with whatever stream_main is
  // doing. Returns the slot index, or -1 if no slot was free, which is always
  // safe: the caller simply gets no prefetch this tick and segment_builder
  // falls back to A_ptr/B_ptr directly.
  int32_t prefetch_adapter(int32_t adapter_id) {
    if (!store_.contains(adapter_id)) return -1;
    const AdapterStore::Entry& e = store_.lookup(adapter_id);
    int slot = staging_.acquire_slot(adapter_id);
    if (slot < 0) return -1;
    const size_t a_bytes = static_cast<size_t>(e.rank) * e.d_model * sizeof(half);
    const size_t b_bytes = static_cast<size_t>(e.out_features) * e.rank * sizeof(half);
    launch_prefetch_copy(reinterpret_cast<const void*>(e.A_ptr), staging_.a_ptr(slot), a_bytes, stream_prefetch_);
    launch_prefetch_copy(reinterpret_cast<const void*>(e.B_ptr), staging_.b_ptr(slot), b_bytes, stream_prefetch_);
    staging_.mark_copy_issued(slot, stream_prefetch_);
    return static_cast<int32_t>(slot);
  }

  bool is_prefetch_ready(int32_t adapter_id) {
    int slot = staging_.find_slot_for_adapter(adapter_id);
    return slot >= 0 && staging_.is_ready(slot);
  }

  void release_prefetch(int32_t adapter_id) { staging_.release_slot(staging_.find_slot_for_adapter(adapter_id)); }

  // GPU-side readback for test_scheduler_bridge.py -- launches a real
  // kernel that dereferences meta_dev_->bridge_ptr, proving the device
  // pointer is actually valid and readable from device code (not just from
  // host-pinned memory, which read_bridge_debug() checks separately).
  std::vector<float> read_bridge_via_kernel() {
    const uint32_t n = meta_host_->bridge_ptr->num_active_adapters;
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_u = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor out_whittle = torch::zeros({static_cast<int64_t>(n)}, opts_f);
    torch::Tensor out_scalars = torch::zeros({3}, opts_u);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_bridge_readback(meta_dev_, reinterpret_cast<float*>(out_whittle.data_ptr()),
                            reinterpret_cast<uint32_t*>(out_scalars.data_ptr()), stream);
    cudaStreamSynchronize(stream);
    auto cpu_w = out_whittle.cpu();
    auto cpu_s = out_scalars.cpu();
    std::vector<float> result(cpu_w.data_ptr<float>(), cpu_w.data_ptr<float>() + n);
    result.push_back(static_cast<float>(cpu_s.data_ptr<int32_t>()[0]));
    result.push_back(static_cast<float>(cpu_s.data_ptr<int32_t>()[1]));
    result.push_back(static_cast<float>(cpu_s.data_ptr<int32_t>()[2]));
    return result;  // [whittle_scores..., num_active_adapters, num_segments, step_id]
  }

  // Debug-only readback for test_metadata_bridge.py -- never used on the
  // live serving path. Returns the bridge fields sliced to num_active_adapters.
  py::dict read_bridge_debug() {
    const SchedulerKernelBridge* b = bridge_.host_ptr_debug();
    const size_t n = b->num_active_adapters;
    py::dict d;
    d["whittle_scores"] = std::vector<float>(b->whittle_scores, b->whittle_scores + n);
    d["t_remaining_ms"] = std::vector<float>(b->t_remaining_ms, b->t_remaining_ms + n);
    d["gwar_pred_next3"] = std::vector<float>(b->gwar_pred_next3, b->gwar_pred_next3 + n);
    d["lambda_hat"] = std::vector<float>(b->lambda_hat, b->lambda_hat + n);
    d["burst_active"] = std::vector<uint8_t>(b->burst_active, b->burst_active + n);
    d["promo_eligible"] = std::vector<uint8_t>(b->promo_eligible, b->promo_eligible + n);
    d["is_hot"] = std::vector<uint8_t>(b->is_hot, b->is_hot + n);
    d["tile_size_code"] = std::vector<uint8_t>(b->tile_size_code, b->tile_size_code + n);
    d["num_active_adapters"] = b->num_active_adapters;
    d["num_segments"] = b->num_segments;
    d["step_id"] = b->step_id;
    d["global_load"] = b->global_load;
    return d;
  }

  void register_adapter(int32_t adapter_id, torch::Tensor A, torch::Tensor B, int32_t rank, float alpha_scale) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "A/B must be fp16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous");
    const int32_t d_model = A.size(-1);
    const int32_t out_features = B.size(-2);
    store_.register_adapter(adapter_id, reinterpret_cast<uintptr_t>(A.data_ptr()),
                             reinterpret_cast<uintptr_t>(B.data_ptr()), rank, d_model, out_features, alpha_scale);
    held_tensors_[adapter_id] = std::make_pair(A, B);
  }

  void evict_adapter(int32_t adapter_id) {
    store_.evict(adapter_id);
    held_tensors_.erase(adapter_id);
  }

  bool rank_supported(int32_t rank) { return warp_pipe_rank_supported(rank); }

  void build_segments(std::vector<int32_t> sorted_adapter_ids, std::vector<int32_t> seg_offsets) {
    build_segment_table(meta_host_, sorted_adapter_ids, seg_offsets, store_, &staging_);
  }

  torch::Tensor shrink(torch::Tensor X, int64_t d_model, int64_t rank, int64_t total_tokens) {
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat16 && X.is_contiguous(), "X must be contiguous fp16 CUDA");
    auto H = torch::zeros({total_tokens, rank}, X.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_bgmv_shrink(reinterpret_cast<const half*>(X.data_ptr()), reinterpret_cast<half*>(H.data_ptr()),
                          meta_dev_, static_cast<int>(d_model), static_cast<int>(rank), stream);
    return H;
  }

  void expand(torch::Tensor H, torch::Tensor Y, int64_t out_features, int64_t rank) {
    TORCH_CHECK(H.is_cuda() && Y.is_cuda(), "H/Y must be CUDA tensors");
    TORCH_CHECK(H.dtype() == torch::kFloat16 && Y.dtype() == torch::kFloat16, "H/Y must be fp16");
    TORCH_CHECK(H.is_contiguous() && Y.is_contiguous(), "H/Y must be contiguous");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_bgmv_expand(reinterpret_cast<const half*>(H.data_ptr()), reinterpret_cast<half*>(Y.data_ptr()),
                          meta_dev_, static_cast<int>(out_features), static_cast<int>(rank), stream);
  }

  // Once-per-step hook: builds ONLY
  // segment boundaries (adapter_id/token_start/token_count), entirely
  // on-device, zero host sync, fixed grid=(1,1,1) -- same graph-safety
  // properties as the old per-call build_segments_kernel path, but called
  // ONCE per decode step from PunicaWrapper.update_metadata()
  // (lora_warp_pipe/punica_wrapper.py), not once per LoRA
  // layer. add_lora_packed() below now assumes this already ran for the
  // current step's token_lora_indices and does not rebuild boundaries
  // itself. Returns false (caller must fall back to the stock path for the
  // whole step) only if total_tokens exceeds WP_MAX_SEGMENTS -- would only
  // happen if max_num_seqs were configured above 256, not the case in any
  // config this project uses.
  bool build_seg_bounds(torch::Tensor token_lora_indices) {
    TORCH_CHECK(token_lora_indices.is_cuda(), "build_seg_bounds: token_lora_indices must be a CUDA tensor");
    torch::Tensor indices = token_lora_indices.to(torch::kInt64).contiguous();
    const int64_t total_tokens = indices.size(0);
    if (total_tokens > WP_MAX_SEGMENTS) return false;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dispatch_build_seg_bounds(indices.data_ptr<int64_t>(), static_cast<int>(total_tokens), meta_dev_, stream);
    return true;
  }

  // Live-serving entry point: mirrors vLLM's own PunicaWrapper.add_lora()
  // shape (full stacked wa_t_all/wb_t_all in, no persistent registration) --
  // see lora_warp_pipe/punica_wrapper.py for the call site.
  // Takes no token_lora_indices and does NOT build the segment table:
  // build_seg_bounds() above must already have run for this step (once, from
  // update_metadata(), before any add_lora() call). This method only resolves
  // THIS layer's
  // A_ptr/B_ptr (wa_t_all/wb_t_all are layer-specific; the segment
  // boundaries are not) and dispatches the DSAK shrink/expand kernels, which
  // read adapter_id/token_start/token_count from meta_dev_ but take
  // wa_base/wb_base/scale/d_model/out_features as direct arguments -- see
  // warp_pipe_r32.cu's bgmv_shrink_dsak_kernel/bgmv_expand_dsak_kernel.
  //
  // Still returns false (caller falls back to the stock path) if rank != 32
  // -- the one precondition this kernel build genuinely cannot relax.
  //
  // Deliberately does NOT use AdapterStore/build_segment_table: those cache
  // device pointers under an adapter_id key for the long-lived
  // scheduler-bridge/benchmark-harness path, which would go stale here since
  // vLLM can reassign which physical adapter occupies a given lora_idx slot
  // between requests. Pointers are recomputed straight from wa_t_all/wb_t_all
  // every call instead (on-device, inside the dsak kernels), the same way
  // vLLM's own bgmv kernels read them.
  bool add_lora_packed(torch::Tensor y, torch::Tensor x, torch::Tensor wa_t_all, torch::Tensor wb_t_all,
                        double scale) {
    const int64_t rank = wb_t_all.size(-1);
    if (rank != 32) return false;
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "add_lora_packed: x/y must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat16 && y.dtype() == torch::kFloat16, "add_lora_packed: x/y must be fp16");

    torch::Tensor x2d = x.view({-1, x.size(-1)});
    torch::Tensor y2d = y.view({-1, y.size(-1)});
    TORCH_CHECK(x2d.is_contiguous() && y2d.is_contiguous(), "add_lora_packed: x/y must be contiguous");
    const int64_t total_tokens = x2d.size(0);
    const int64_t d_model = x2d.size(1);
    const int64_t out_features = y2d.size(1);
    TORCH_CHECK(wa_t_all.is_contiguous() && wb_t_all.is_contiguous(),
                "add_lora_packed: wa_t_all/wb_t_all must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // add_lora_packed() is called ~80x per decode step (once per LoRA-enabled
    // linear layer), so H is a reused buffer rather than a fresh
    // torch::zeros({total_tokens, rank}) per call, which would be a separate
    // cudaMalloc-from-pool plus a memset plus their launch overhead, 80 times a step.
    // Dropping the zero-init is safe: bgmv_expand_dsak_kernel only ever reads H rows
    // inside a real adapter segment (built by build_seg_bounds_kernel from
    // token_lora_indices>=0 runs), and rows for no-adapter tokens are never read by
    // anything, so stale data left in them cannot affect correctness. H_buf_ is grown (never shrunk) on
    // demand and reused across calls -- allocates at most a handful of
    // times across a whole server lifetime instead of every single call.
    if (!H_buf_.defined() || H_buf_.size(0) < total_tokens || H_buf_.size(1) != rank ||
        H_buf_.scalar_type() != x2d.scalar_type() || H_buf_.device() != x2d.device()) {
      H_buf_ = torch::empty({std::max<int64_t>(total_tokens, WP_MAX_SEGMENTS), rank}, x2d.options());
    }
    torch::Tensor H = H_buf_.narrow(0, 0, total_tokens);
    dispatch_bgmv_shrink_dsak(reinterpret_cast<const half*>(x2d.data_ptr()), reinterpret_cast<half*>(H.data_ptr()),
                               meta_dev_, reinterpret_cast<const half*>(wa_t_all.data_ptr()), wa_t_all.stride(0),
                               static_cast<float>(scale), static_cast<int>(d_model), static_cast<int>(rank), stream);
    dispatch_bgmv_expand_dsak(reinterpret_cast<const half*>(H.data_ptr()), reinterpret_cast<half*>(y2d.data_ptr()),
                               meta_dev_, reinterpret_cast<const half*>(wb_t_all.data_ptr()), wb_t_all.stride(0),
                               static_cast<int>(out_features), static_cast<int>(rank), stream);
    return true;
  }

  // SchedulerBridge.write() calls this once per scheduler tick. All array args are
  // CPU float32/uint8 tensors of length num_active_adapters, in the same adapter-index
  // order build_segments' sorted_adapter_ids uses (see AdapterIdMap in
  // lora_warp_pipe/scheduler_bridge.py).
  void write_bridge(torch::Tensor whittle_scores, torch::Tensor t_remaining_ms, torch::Tensor gwar_pred_next3,
                     torch::Tensor lambda_hat, torch::Tensor burst_active, torch::Tensor promo_eligible,
                     torch::Tensor is_hot, torch::Tensor tile_size_code, int64_t num_active_adapters,
                     int64_t num_segments, int64_t step_id, double global_load) {
    TORCH_CHECK(num_active_adapters <= WP_MAX_ADAPTERS, "write_bridge: num_active_adapters exceeds WP_MAX_ADAPTERS");
    auto check_f32 = [](const torch::Tensor& t, const char* name) {
      TORCH_CHECK(!t.is_cuda() && t.dtype() == torch::kFloat32 && t.is_contiguous(), name,
                  " must be a contiguous CPU float32 tensor");
    };
    auto check_u8 = [](const torch::Tensor& t, const char* name) {
      TORCH_CHECK(!t.is_cuda() && t.dtype() == torch::kUInt8 && t.is_contiguous(), name,
                  " must be a contiguous CPU uint8 tensor");
    };
    check_f32(whittle_scores, "whittle_scores");
    check_f32(t_remaining_ms, "t_remaining_ms");
    check_f32(gwar_pred_next3, "gwar_pred_next3");
    check_f32(lambda_hat, "lambda_hat");
    check_u8(burst_active, "burst_active");
    check_u8(promo_eligible, "promo_eligible");
    check_u8(is_hot, "is_hot");
    check_u8(tile_size_code, "tile_size_code");
    bridge_.write(whittle_scores.data_ptr<float>(), t_remaining_ms.data_ptr<float>(),
                  gwar_pred_next3.data_ptr<float>(), lambda_hat.data_ptr<float>(),
                  burst_active.data_ptr<uint8_t>(), promo_eligible.data_ptr<uint8_t>(), is_hot.data_ptr<uint8_t>(),
                  tile_size_code.data_ptr<uint8_t>(), static_cast<uint32_t>(num_active_adapters),
                  static_cast<uint32_t>(num_segments), static_cast<uint32_t>(step_id),
                  static_cast<float>(global_load));
  }

 private:
  AdapterStore store_;
  StagingBufferPool staging_;
  SchedulerKernelBridgeHost bridge_;
  WarpPipeMetadata* meta_host_;
  WarpPipeMetadata* meta_dev_;
  cudaStream_t stream_prefetch_;
  std::unordered_map<int32_t, std::pair<torch::Tensor, torch::Tensor>> held_tensors_;
  // Reusable intermediate-H scratch buffer for add_lora_packed()'s live
  // path -- see that method's comment. Default-constructed (undefined)
  // until first use; grown on demand, never shrunk.
  torch::Tensor H_buf_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<WarpPipeExtension>(m, "WarpPipeExtension")
      .def(py::init<>())
      .def("register_adapter", &WarpPipeExtension::register_adapter)
      .def("evict_adapter", &WarpPipeExtension::evict_adapter)
      .def("rank_supported", &WarpPipeExtension::rank_supported)
      .def("build_segments", &WarpPipeExtension::build_segments)
      .def("shrink", &WarpPipeExtension::shrink)
      .def("expand", &WarpPipeExtension::expand)
      .def("write_bridge", &WarpPipeExtension::write_bridge)
      .def("build_seg_bounds", &WarpPipeExtension::build_seg_bounds)
      .def("add_lora_packed", &WarpPipeExtension::add_lora_packed)
      .def("prefetch_adapter", &WarpPipeExtension::prefetch_adapter)
      .def("is_prefetch_ready", &WarpPipeExtension::is_prefetch_ready)
      .def("release_prefetch", &WarpPipeExtension::release_prefetch)
      .def("read_bridge_debug", &WarpPipeExtension::read_bridge_debug)
      .def("read_bridge_via_kernel", &WarpPipeExtension::read_bridge_via_kernel);
}
