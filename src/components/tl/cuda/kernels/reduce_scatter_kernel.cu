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

#ifdef __cplusplus
}
#endif

#include "nvls.cuh"

template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    reduce_scatter_kernel_vec32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        const uint32_t total_blocks, uint64_t launch_counter,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t *dst_u32)
{
    // pre barrier
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 1));

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = offset + thread_offset; idx + 3 < offset + count;
         idx += stride) {
        uint4 val;
        NvlsOps::ld(val, base_u32 + idx);
        uint32_t *dst                     = dst_u32 + (idx - offset);
        reinterpret_cast<uint4 *>(dst)[0] = val;
    }

    // post barrier
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 2));
}

template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    reduce_scatter_kernel_scalar32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        const uint32_t total_blocks, uint64_t launch_counter,
        uint32_t *base_u32, size_t offset, size_t count, uint32_t *dst_u32)
{
    // pre barrier
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 1));

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = offset + thread_offset; idx + 3 < offset + count;
         idx += stride) {
        typename NvlsOps::value_type v0, v1, v2, v3;
        NvlsOps::ld(v0, base_u32 + idx + 0);
        NvlsOps::ld(v1, base_u32 + idx + 1);
        NvlsOps::ld(v2, base_u32 + idx + 2);
        NvlsOps::ld(v3, base_u32 + idx + 3);
        dst_u32[idx - offset + 0] = v0;
        dst_u32[idx - offset + 1] = v1;
        dst_u32[idx - offset + 2] = v2;
        dst_u32[idx - offset + 3] = v3;
    }

    // post barrier
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 2));
}

template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    reduce_scatter_kernel_scalar64(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        const uint32_t total_blocks, uint64_t launch_counter,
        uint64_t *base_u64, size_t offset, size_t count, uint64_t *dst_u64)
{
    // pre barrier
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 1));

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    size_t stride        = blockDim.x * gridDim.x * 2;

    for (size_t idx = offset + thread_offset; idx + 1 < offset + count;
         idx += stride) {
        typename NvlsOps::value_type v0, v1;
        NvlsOps::ld(v0, base_u64 + idx + 0);
        NvlsOps::ld(v1, base_u64 + idx + 1);
        dst_u64[idx - offset + 0] = v0;
        dst_u64[idx - offset + 1] = v1;
    }

    // post barrier
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 2));
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_reduce_scatter_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr dst_ptr, CUdeviceptr mc_base_addr, CUdeviceptr mc_control_addr,
    CUdeviceptr uc_control_addr, uint64_t launch_counter, size_t offset,
    size_t count, ucc_datatype_t datatype, uint32_t tsize)
{
    ucc_assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    ucc_assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    /* NVLS requires 16-byte alignment */
    ucc_assert(offset % 4 == 0);
    ucc_assert(count % 4 == 0);
    ucc_assert(mc_base_addr % 16 == 0);
    ucc_assert(mc_control_addr % 16 == 0);

    uint32_t *base_u32 = reinterpret_cast<uint32_t *>(mc_base_addr);
    ucc_tl_cuda_nvls_control_t
        *mc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(
            mc_control_addr);
    ucc_tl_cuda_nvls_control_t
        *uc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(
            uc_control_addr);
    // total num of blocks in the multicast group, num gpus * num blocks per gpu, used for barrier synchronization
    uint32_t expected_blocks = sm_count * tsize;

    switch (datatype) {
    case UCC_DT_FLOAT32:
        reduce_scatter_kernel_vec32<NvlsFp32Ops>
            <<<sm_count, threads, 0, stream>>>(
                mc_bar,
                uc_bar,
                expected_blocks,
                launch_counter,
                base_u32,
                offset,
                count,
                reinterpret_cast<uint32_t *>(dst_ptr));
        break;
    case UCC_DT_BFLOAT16:
        reduce_scatter_kernel_vec32<NvlsBf16Ops>
            <<<sm_count, threads, 0, stream>>>(
                mc_bar,
                uc_bar,
                expected_blocks,
                launch_counter,
                base_u32,
                offset,
                count,
                reinterpret_cast<uint32_t *>(dst_ptr));
        break;
    case UCC_DT_INT32:
        reduce_scatter_kernel_scalar32<NvlsInt32Ops>
            <<<sm_count, threads, 0, stream>>>(
                mc_bar,
                uc_bar,
                expected_blocks,
                launch_counter,
                base_u32,
                offset,
                count,
                reinterpret_cast<uint32_t *>(dst_ptr));
        break;
    case UCC_DT_UINT32:
        reduce_scatter_kernel_scalar32<NvlsUint32Ops>
            <<<sm_count, threads, 0, stream>>>(
                mc_bar,
                uc_bar,
                expected_blocks,
                launch_counter,
                base_u32,
                offset,
                count,
                reinterpret_cast<uint32_t *>(dst_ptr));
        break;
    case UCC_DT_INT64:
        {
            uint64_t *base_u64 = reinterpret_cast<uint64_t *>(mc_base_addr);
            size_t offset_u64 = offset / 2;
            size_t count_u64 = count / 2;
            reduce_scatter_kernel_scalar64<NvlsInt64Ops>
                <<<sm_count, threads, 0, stream>>>(
                    mc_bar,
                    uc_bar,
                    expected_blocks,
                    launch_counter,
                    base_u64,
                    offset_u64,
                    count_u64,
                    reinterpret_cast<uint64_t *>(dst_ptr));
        }
        break;
    case UCC_DT_UINT64:
        {
            uint64_t *base_u64 = reinterpret_cast<uint64_t *>(mc_base_addr);
            size_t offset_u64 = offset / 2;
            size_t count_u64 = count / 2;
            reduce_scatter_kernel_scalar64<NvlsUint64Ops>
                <<<sm_count, threads, 0, stream>>>(
                    mc_bar,
                    uc_bar,
                    expected_blocks,
                    launch_counter,
                    base_u64,
                    offset_u64,
                    count_u64,
                    reinterpret_cast<uint64_t *>(dst_ptr));
        }
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
