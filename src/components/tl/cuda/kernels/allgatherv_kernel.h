/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_ALLGATHERV_KERNEL_H_
#define UCC_TL_CUDA_ALLGATHERV_KERNEL_H_

#include <cuda.h>
#include "ucc/api/ucc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel function declaration */
ucc_status_t post_allgatherv_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr src_ptr, CUdeviceptr mc_base_addr, size_t my_offset,
    size_t my_count, CUdeviceptr mc_control_addr, CUdeviceptr uc_control_addr,
    uint64_t launch_counter, uint32_t tsize);

#ifdef __cplusplus
}
#endif

#endif // UCC_TL_CUDA_ALLGATHERV_KERNEL_H_
