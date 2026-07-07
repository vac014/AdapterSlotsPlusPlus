#include "warp_pipe_dispatcher.h"

#include <stdexcept>

#include "../kernels/warp_pipe_r32.h"
#include "../kernels/warp_pipe_r8.h"
#include "../kernels/warp_pipe_r16.h"
#include "../kernels/warp_pipe_r64.h"

// r8/r16/r64 (Task #109): cp.async pipeline shrink/expand only, generalized
// from warp_pipe_r32.cu via warp_pipe_rgen.cuh's template<int RANK>.
// build_segments/build_seg_bounds/DSAK stay rank=32-only below -- DSAK's
// small-segment path structurally depends on RANK==warpSize==32 (see
// warp_pipe_rgen.cuh's header comment), and dispatch_build_segments is dead
// code anyway (build_segments() at the Python binding layer goes through
// segment_builder.cpp's build_segment_table, already rank-agnostic, never
// through this function -- see warp_pipe_bindings.cpp).
bool warp_pipe_rank_supported(int rank) { return rank == 8 || rank == 16 || rank == 32 || rank == 64; }

void dispatch_build_segments(const int64_t* token_lora_indices, int total_tokens,
                              const half* wa_base, int64_t wa_stride0,
                              const half* wb_base, int64_t wb_stride0,
                              float alpha_scale, int rank, int d_model, int out_features,
                              WarpPipeMetadata* meta, cudaStream_t stream) {
  if (rank != 32) {
    throw std::runtime_error("dispatch_build_segments: only rank=32 supported (dead code path, see comment above)");
  }
  launch_build_segments_r32(token_lora_indices, total_tokens, wa_base, wa_stride0,
                             wb_base, wb_stride0, alpha_scale, rank, d_model, out_features,
                             meta, stream);
}

void dispatch_bgmv_shrink(const half* X, half* H, const WarpPipeMetadata* meta, int d_model, int rank,
                           cudaStream_t stream) {
  switch (rank) {
    case 8:  launch_bgmv_shrink_r8(X, H, meta, d_model, stream); return;
    case 16: launch_bgmv_shrink_r16(X, H, meta, d_model, stream); return;
    case 32: launch_bgmv_shrink_r32(X, H, meta, d_model, stream); return;
    case 64: launch_bgmv_shrink_r64(X, H, meta, d_model, stream); return;
    default: throw std::runtime_error("dispatch_bgmv_shrink: unsupported rank (only 8/16/32/64)");
  }
}

void dispatch_bgmv_expand(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features, int rank,
                           cudaStream_t stream) {
  switch (rank) {
    case 8:  launch_bgmv_expand_r8(H, Y, meta, out_features, stream); return;
    case 16: launch_bgmv_expand_r16(H, Y, meta, out_features, stream); return;
    case 32: launch_bgmv_expand_r32(H, Y, meta, out_features, stream); return;
    case 64: launch_bgmv_expand_r64(H, Y, meta, out_features, stream); return;
    default: throw std::runtime_error("dispatch_bgmv_expand: unsupported rank (only 8/16/32/64)");
  }
}

void dispatch_build_seg_bounds(const int64_t* token_lora_indices, int total_tokens, WarpPipeMetadata* meta,
                                cudaStream_t stream) {
  launch_build_seg_bounds_r32(token_lora_indices, total_tokens, meta, stream);
}

void dispatch_bgmv_shrink_dsak(const half* X, half* H, const WarpPipeMetadata* meta, const half* wa_base,
                                int64_t wa_stride0, float alpha_scale, int d_model, int rank, cudaStream_t stream) {
  if (rank != 32) {
    throw std::runtime_error("dispatch_bgmv_shrink_dsak: only rank=32 is supported");
  }
  launch_bgmv_shrink_dsak_r32(X, H, meta, wa_base, wa_stride0, alpha_scale, d_model, stream);
}

void dispatch_bgmv_expand_dsak(const half* H, half* Y, const WarpPipeMetadata* meta, const half* wb_base,
                                int64_t wb_stride0, int out_features, int rank, cudaStream_t stream) {
  if (rank != 32) {
    throw std::runtime_error("dispatch_bgmv_expand_dsak: only rank=32 is supported");
  }
  launch_bgmv_expand_dsak_r32(H, Y, meta, wb_base, wb_stride0, out_features, stream);
}
