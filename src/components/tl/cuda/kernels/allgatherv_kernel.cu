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

__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allgatherv_kernel_vec32(
        ucc_tl_cuda_nvls_control_t *mc_bar, ucc_tl_cuda_nvls_control_t *uc_bar,
        /* block count per gpu * num gpus in Multicast group */
        const uint32_t total_blocks,
        uint64_t launch_counter, uint32_t *src_u32, uint32_t *base_u32,
        size_t my_offset, size_t my_count)
{
    // Pre-barrier: ensure all ranks are ready before writing
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 1));

    // Each rank copies its data to NVLS mc buffer at its specific offset using multimem store
    // Datatype agnostic - just copy raw data as uint4 vectors (16 bytes at a time)
    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = thread_offset; idx < my_count; idx += stride) {
        // Read 4 uint32_t values (16 bytes) from source
        uint4     val = reinterpret_cast<uint4 *>(src_u32)[idx / 4];
        // Write to destination at my_offset using multimem store (datatype agnostic)
        uint32_t *dst = base_u32 + my_offset + idx;
        MULTIMEM_ST_U32(val, dst);
    }

    // Post-barrier: ensure all ranks have completed writing before reading
    nvls_bar(
        &(mc_bar->arrival_counter),
        &(uc_bar->arrival_counter),
        total_blocks * (launch_counter * 2 + 2));
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_allgatherv_kernel(
    cudaStream_t stream, uint32_t sm_count, uint32_t threads,
    CUdeviceptr src_ptr, CUdeviceptr mc_base_addr, size_t my_offset,
    size_t my_count, CUdeviceptr mc_control_addr, CUdeviceptr uc_control_addr,
    uint64_t launch_counter, uint32_t tsize)
{
    ucc_assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    ucc_assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    /* NVLS requires 16-byte alignment */
    ucc_assert(my_offset % 4 == 0);
    ucc_assert(my_count % 4 == 0);
    ucc_assert(mc_base_addr % 16 == 0);
    ucc_assert(mc_control_addr % 16 == 0);
    ucc_assert(src_ptr % 16 == 0);

    uint32_t *src_u32  = reinterpret_cast<uint32_t *>(src_ptr);
    uint32_t *base_u32 = reinterpret_cast<uint32_t *>(mc_base_addr);
    ucc_tl_cuda_nvls_control_t
        *mc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(
            mc_control_addr);
    ucc_tl_cuda_nvls_control_t
        *uc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(
            uc_control_addr);
    /* total num of blocks in the multicast group, used for barrier sync */
    uint32_t expected_blocks = sm_count * tsize;

    allgatherv_kernel_vec32<<<sm_count, threads, 0, stream>>>(
        mc_bar,
        uc_bar,
        expected_blocks,
        launch_counter,
        src_u32,
        base_u32,
        my_offset,
        my_count);

    CUDA_CHECK(cudaGetLastError());

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
