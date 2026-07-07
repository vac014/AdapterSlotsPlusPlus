#include "warp_pipe_r64.h"
#include "warp_pipe_rgen.cuh"

void launch_bgmv_shrink_r64(const half* X, half* H, const WarpPipeMetadata* meta, int d_model,
                               cudaStream_t stream) {
  wp_rgen::launch_bgmv_shrink<64>(X, H, meta, d_model, stream);
}

void launch_bgmv_expand_r64(const half* H, half* Y, const WarpPipeMetadata* meta, int out_features,
                               cudaStream_t stream) {
  wp_rgen::launch_bgmv_expand<64>(H, Y, meta, out_features, stream);
}
