/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_REDUCE_SCATTER_KERNEL_H_
#define UCC_TL_CUDA_REDUCE_SCATTER_KERNEL_H_

#include <cuda.h>
#include "ucc/api/ucc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kernel function declaration
ucc_status_t post_reduce_scatter_kernel(cudaStream_t stream, uint32_t sm_count,
                                        uint32_t threads, CUdeviceptr src_addr,
                                        CUdeviceptr dst_addr,
                                        size_t src_size_bytes, uint32_t rank,
                                        uint32_t tsize);
#ifdef __cplusplus
}
#endif

#endif // UCC_TL_CUDA_REDUCE_SCATTER_KERNEL_H_
