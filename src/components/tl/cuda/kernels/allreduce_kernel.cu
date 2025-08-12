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

#ifdef __cplusplus
}
#endif

#include "nvls.cuh"

// vectorized allreduce kernel for 32-bit lanes
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
allreduce_kernel_vec32(uint32_t *base_u32, size_t count_u32, uint32_t rank,
                       uint32_t tsize)
{
    size_t chunk_start = ((int64_t)count_u32 * (int64_t)rank) / (int64_t)tsize;
    size_t chunk_end   = ((int64_t)count_u32 * (int64_t)(rank + 1)) / (int64_t)tsize;

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = chunk_start + thread_offset; idx < chunk_end; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, base_u32 + idx);
        NvlsOps::st(val, base_u32 + idx);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_allreduce_kernel(cudaStream_t stream, uint32_t sm_count,
                                   uint32_t threads, CUdeviceptr src_addr,
                                   size_t src_size_bytes, uint32_t rank,
                                   uint32_t tsize, ucc_datatype_t datatype)
{
    assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);
    uint32_t *base_u32   = reinterpret_cast<uint32_t *>(src_addr);
    size_t    count_u32  = src_size_bytes / sizeof(uint32_t);

    switch (datatype) {
    case UCC_DT_FLOAT32:
        assert(((uintptr_t)(src_addr) % 8) == 0);
        allreduce_kernel_vec32<NvlsFp32Ops><<<sm_count, threads, 0, stream>>>(
            base_u32, count_u32, rank, tsize);
        break;
    case UCC_DT_BFLOAT16:
        assert(((uintptr_t)(src_addr) % 8) == 0);
        allreduce_kernel_vec32<NvlsBf16Ops><<<sm_count, threads, 0, stream>>>(
            base_u32, count_u32, rank, tsize);
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
