/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "utils/arch/cuda_def.h"
#include "../tl_cuda.h"

#include "nvls.cuh"

__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    reduce_scatter_kernel(float *src_addr, float *dst_addr, size_t src_count,
                          uint32_t rank, uint32_t tsize)
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
        float *dst                         = dst_addr + (idx - chunk_start);
        reinterpret_cast<float4 *>(dst)[0] = *reinterpret_cast<float4 *>(&val);
    }

    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_reduce_scatter_kernel(cudaStream_t stream, uint32_t sm_count,
                                        uint32_t threads, CUdeviceptr src_addr,
                                        CUdeviceptr dst_addr,
                                        size_t src_size_bytes, uint32_t rank,
                                        uint32_t tsize)
{
    assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);
    reduce_scatter_kernel<<<sm_count, threads, 0, stream>>>(
        (float *)src_addr, (float *)dst_addr, src_size_bytes / sizeof(float),
        rank, tsize);
    CUDA_CHECK(cudaGetLastError());

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
