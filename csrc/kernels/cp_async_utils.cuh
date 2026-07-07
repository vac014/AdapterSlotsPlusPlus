#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// cp.async wrappers.
//
// A 128-byte `cp.async.cg.shared.global [%0], [%1], 128;` is the shape you
// want here and it is ILLEGAL PTX: cp.async.cg only accepts {4,8,16}-byte
// operands per instruction. So each thread issues its own 16-byte (8x fp16)
// cp.async, and the producer warps cooperatively cover a full tile between
// them.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* gmem_src) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

// Cooperative tiled load: a group of `group_size` threads (a whole number of
// warps) jointly copies a [num_rows, tile_cols] row-major fp16 tile from
// global memory into shared memory.
//
// `src_row_stride` is the REAL row stride of the source matrix in elements
// (d_model for an A matrix, rank for a B matrix), NOT tile_cols. Passing
// tile_cols lands every row at the wrong global offset the moment
// src_row_stride != tile_cols, which is exactly what happens when d_model is
// tiled into TILE_K-wide chunks. `dst_row_stride` is the padded shared-memory
// stride (tile_cols + WP_SMEM_BANK_PAD), which avoids bank conflicts.
__device__ __forceinline__ void load_tile_async(
    half* smem_dst, const half* gmem_src,
    int num_rows, int tile_cols,
    int src_row_stride, int dst_row_stride,
    int tid_in_group, int group_size) {
  constexpr int ELEMS_PER_THREAD = 8;  // 16 bytes = 8 fp16 elements
  const int total_elems = num_rows * tile_cols;
  for (int base = tid_in_group * ELEMS_PER_THREAD; base < total_elems;
       base += group_size * ELEMS_PER_THREAD) {
    const int row = base / tile_cols;
    const int col = base % tile_cols;
    half* dst = smem_dst + row * dst_row_stride + col;
    const half* src = gmem_src + row * src_row_stride + col;
    cp_async_cg_16(dst, src);
  }
}
