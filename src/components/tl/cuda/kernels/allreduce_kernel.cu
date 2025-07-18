/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "utils/arch/cuda_def.h"
#include "../tl_cuda.h"
#include "ucc/api/ucc.h"

#include "nvls.cuh"

#ifdef __cplusplus
}
#endif

#include <cuda_bf16.h>


#define MAX_THREADS 1024
#define MAX_BLOCKS 4

__global__ void __launch_bounds__(MAX_THREADS)
    allreduce_kernel_fp32(float *src_addr, size_t src_count, uint32_t rank,
                     uint32_t tsize)
{
    size_t chunk_start = ((int64_t)src_count * (int64_t)rank) / (int64_t)tsize;
    size_t chunk_end =
        ((int64_t)src_count * (int64_t)(rank + 1)) / (int64_t)tsize;

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = chunk_start + thread_offset; idx < chunk_end;
         idx += stride) {
        uint4 val;
        MULTIMEM_LD(val, src_addr + idx);
        MULTIMEM_ST(val, src_addr + idx);
    }

    return;
}

__global__ void __launch_bounds__(MAX_THREADS)
    allreduce_kernel_bfloat16(nv_bfloat16 *src_addr, size_t src_count, uint32_t rank,
                     uint32_t tsize)
{
    size_t chunk_start = ((int64_t)src_count * (int64_t)rank) / (int64_t)tsize;
    size_t chunk_end =
        ((int64_t)src_count * (int64_t)(rank + 1)) / (int64_t)tsize;

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = chunk_start + thread_offset; idx < chunk_end;
         idx += stride) {
        uint4 val;
        MULTIMEM_LD_BF16(val, src_addr + idx);
        MULTIMEM_ST_BF16(val, src_addr + idx);
    }

    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_allreduce_kernel(cudaStream_t stream, CUdeviceptr src_addr,
                                   size_t src_size_bytes, uint32_t rank,
                                   uint32_t tsize, ucc_datatype_t datatype)
{
    switch (datatype) {
    case UCC_DT_FLOAT32:
        allreduce_kernel_fp32<<<MAX_BLOCKS, MAX_THREADS, 0, stream>>>(
            (float *)src_addr, src_size_bytes / sizeof(float),
            rank, tsize);
        break;
    case UCC_DT_BFLOAT16:
        allreduce_kernel_bfloat16<<<MAX_BLOCKS, MAX_THREADS, 0, stream>>>(
            (nv_bfloat16 *)src_addr, src_size_bytes / sizeof(nv_bfloat16),
            rank, tsize);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
