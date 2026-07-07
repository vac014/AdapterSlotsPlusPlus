#include "warp_pipe_r8.h"
#include "warp_pipe_rgen.cuh"

void launch_bgmv_shrink_r8(const half* X, half* H, const WarpPipeMetadata* meta, int d_model,
                               cudaStream_t stream) {
  wp_rgen::launch_bgmv_shrink<8>(X, H, meta, d_model, stream);
}

void launch_bgmv_expand_r8(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features,
                               cudaStream_t stream) {
  wp_rgen::launch_bgmv_expand<8>(H, Y, meta, out_features, stream);
}
