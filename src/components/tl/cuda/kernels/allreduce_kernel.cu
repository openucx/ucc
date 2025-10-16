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


// vectorized allreduce kernel for 32-bit lanes
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allreduce_kernel_vec32(ucc_tl_cuda_nvls_control_t *mc_bar,
                           ucc_tl_cuda_nvls_control_t *uc_bar,
                           const uint32_t total_blocks, // block count per gpu * num gpus in Multicast group
                           uint64_t launch_counter,
                           uint32_t *base_u32, size_t count_u32, uint32_t rank,
                           uint32_t tsize)
{
    // pre barrier
    nvls_bar(&(mc_bar->arrival_counter), &(uc_bar->arrival_counter), total_blocks * (launch_counter * 2 + 1));

    // Kernel execution
    size_t chunk_start = ((int64_t)count_u32 * (int64_t)rank) / (int64_t)tsize;
    size_t chunk_end   = ((int64_t)count_u32 * (int64_t)(rank + 1)) / (int64_t)tsize;

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    for (size_t idx = chunk_start + thread_offset; idx < chunk_end; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, base_u32 + idx);
        NvlsOps::st(val, base_u32 + idx);
    }

    // post barrier
    nvls_bar(&(mc_bar->arrival_counter), &(uc_bar->arrival_counter), total_blocks * (launch_counter * 2 + 2));
}

// DEDICATED BARRIER KERNELS: Single-thread kernels for synchronization only

// Pre-barrier: Wait for all GPUs to finish input copy
__global__ void nvls_pre_barrier_kernel(ucc_tl_cuda_nvls_control_t *mc_bar,
                                        ucc_tl_cuda_nvls_control_t *uc_bar,
                                        uint64_t expected_count)
{
    // Only thread 0 in block 0 participates - minimal contention (8 threads total across all GPUs)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvls_bar(&(mc_bar->arrival_counter), &(uc_bar->arrival_counter), expected_count);
    }
}

// Post-barrier: Wait for all GPUs to finish computation
__global__ void nvls_post_barrier_kernel(ucc_tl_cuda_nvls_control_t *mc_bar,
                                         ucc_tl_cuda_nvls_control_t *uc_bar,
                                         uint64_t expected_count)
{
    // Only thread 0 in block 0 participates - minimal contention (8 threads total across all GPUs)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvls_bar(&(mc_bar->arrival_counter), &(uc_bar->arrival_counter), expected_count);
    }
}

// PURE COMPUTE KERNEL: No barriers, maximum performance
template <typename NvlsOps>
__global__ void __launch_bounds__(UCC_TL_CUDA_MAX_NVLS_THREADS)
    allreduce_kernel_vec32_no_barriers(uint32_t *base_u32, size_t count_u32,
                                      uint32_t rank, uint32_t tsize)
{
    // NO BARRIERS - Pure computation only

    // Work distribution per rank (same as original)
    size_t chunk_start = ((int64_t)count_u32 * (int64_t)rank) / (int64_t)tsize;
    size_t chunk_end   = ((int64_t)count_u32 * (int64_t)(rank + 1)) / (int64_t)tsize;

    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride        = blockDim.x * gridDim.x * 4;

    // Pure NVLS computation - hardware handles atomicity
    for (size_t idx = chunk_start + thread_offset; idx < chunk_end; idx += stride) {
        uint4 val;
        NvlsOps::ld(val, base_u32 + idx);  // Hardware-atomic load-reduce
        NvlsOps::st(val, base_u32 + idx);  // Hardware-atomic store
    }

    // NO BARRIERS - Finish immediately when computation done
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t post_allreduce_kernel(cudaStream_t stream, uint32_t sm_count,
                                   uint32_t threads, CUdeviceptr mc_base_addr,
                                   size_t src_size_bytes,
                                   CUdeviceptr mc_control_addr,
                                   CUdeviceptr uc_control_addr,
                                   uint64_t launch_counter,
                                   uint32_t rank,
                                   uint32_t tsize, ucc_datatype_t datatype)
{
    assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);
    uint32_t *base_u32   = reinterpret_cast<uint32_t *>(mc_base_addr);
    size_t    count_u32  = src_size_bytes / sizeof(uint32_t);
    ucc_tl_cuda_nvls_control_t *mc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(mc_control_addr);
    ucc_tl_cuda_nvls_control_t *uc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(uc_control_addr);
    uint32_t expected_blocks = sm_count * tsize; // total num of blocks in the multicast group, num gpus * num blocks per gpu, used for barrier synchronization

    switch (datatype) {
    case UCC_DT_FLOAT32:
        assert(((uintptr_t)(mc_base_addr) % 8) == 0);
        allreduce_kernel_vec32<NvlsFp32Ops><<<sm_count, threads, 0, stream>>>(
            mc_bar, uc_bar, expected_blocks, launch_counter, base_u32, count_u32, rank, tsize);
        break;
    case UCC_DT_BFLOAT16:
        assert(((uintptr_t)(mc_base_addr) % 8) == 0);
        allreduce_kernel_vec32<NvlsBf16Ops><<<sm_count, threads, 0, stream>>>(
            mc_bar, uc_bar, expected_blocks, launch_counter, base_u32, count_u32, rank, tsize);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());

    return UCC_OK;
}

// NEW: Dedicated barrier kernel launcher - separates sync from compute
ucc_status_t post_allreduce_kernel_dedicated_barriers(cudaStream_t stream, uint32_t sm_count,
                                                     uint32_t threads, CUdeviceptr mc_base_addr,
                                                     size_t src_size_bytes,
                                                     CUdeviceptr mc_control_addr,
                                                     CUdeviceptr uc_control_addr,
                                                     uint64_t launch_counter,
                                                     uint32_t rank, uint32_t tsize,
                                                     ucc_datatype_t datatype)
{
    assert(sm_count > 0 && sm_count <= UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
    assert(threads > 0 && threads <= UCC_TL_CUDA_MAX_NVLS_THREADS);

    uint32_t *base_u32 = reinterpret_cast<uint32_t *>(mc_base_addr);
    size_t count_u32 = src_size_bytes / sizeof(uint32_t);
    ucc_tl_cuda_nvls_control_t *mc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(mc_control_addr);
    ucc_tl_cuda_nvls_control_t *uc_bar = reinterpret_cast<ucc_tl_cuda_nvls_control_t *>(uc_control_addr);

    // Only 8 threads total (1 per GPU) participate in barriers - minimal contention
    uint64_t pre_barrier_count = tsize * (launch_counter * 2 + 1);
    uint64_t post_barrier_count = tsize * (launch_counter * 2 + 2);

    // PHASE 1: Pre-barrier - wait for all GPUs to finish input copy
    nvls_pre_barrier_kernel<<<1, 1, 0, stream>>>(mc_bar, uc_bar, pre_barrier_count);
    CUDA_CHECK(cudaGetLastError());

    // PHASE 2: Pure compute kernel - no barriers, maximum performance
    switch (datatype) {
    case UCC_DT_FLOAT32:
        assert(((uintptr_t)(mc_base_addr) % 8) == 0);
        allreduce_kernel_vec32_no_barriers<NvlsFp32Ops><<<sm_count, threads, 0, stream>>>(
            base_u32, count_u32, rank, tsize);
        break;
    case UCC_DT_BFLOAT16:
        assert(((uintptr_t)(mc_base_addr) % 8) == 0);
        allreduce_kernel_vec32_no_barriers<NvlsBf16Ops><<<sm_count, threads, 0, stream>>>(
            base_u32, count_u32, rank, tsize);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());

    // PHASE 3: Post-barrier - wait for all GPUs to finish computation
    nvls_post_barrier_kernel<<<1, 1, 0, stream>>>(mc_bar, uc_bar, post_barrier_count);
    CUDA_CHECK(cudaGetLastError());

    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
