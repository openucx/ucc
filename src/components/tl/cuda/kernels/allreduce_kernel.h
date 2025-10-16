/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_ALLREDUCE_KERNEL_H_
#define UCC_TL_CUDA_ALLREDUCE_KERNEL_H_

#include <cuda.h>
#include "ucc/api/ucc.h"

#ifdef __cplusplus
extern "C" {
#endif

// Original kernel with internal barriers
ucc_status_t post_allreduce_kernel(cudaStream_t stream, uint32_t sm_count,
                                   uint32_t threads, CUdeviceptr mc_base_addr,
                                   size_t src_size_bytes,
                                   CUdeviceptr mc_control_addr,
                                   CUdeviceptr uc_control_addr,
                                   uint64_t launch_counter, // launch counter for specific NVLS task in flight slot, used for barrier synchronization
                                   uint32_t rank,
                                   uint32_t tsize, ucc_datatype_t datatype);

// NEW: Dedicated barrier kernels approach - separates synchronization from computation
// Architecture: pre_barrier<<<1,1>>> → compute<<<sm_count,threads>>> → post_barrier<<<1,1>>>
// Benefits:
//   - Eliminates 32-block barrier contention (8 threads total vs 100s of threads)
//   - Enables higher SM counts without synchronization overhead
//   - Consistent ~17μs compute latency (pure computation time)
//   - Maintains correctness (proper copy→compute→copy synchronization)
ucc_status_t post_allreduce_kernel_dedicated_barriers(cudaStream_t stream, uint32_t sm_count,
                                                     uint32_t threads, CUdeviceptr mc_base_addr,
                                                     size_t src_size_bytes,
                                                     CUdeviceptr mc_control_addr,
                                                     CUdeviceptr uc_control_addr,
                                                     uint64_t launch_counter,
                                                     uint32_t rank, uint32_t tsize,
                                                     ucc_datatype_t datatype);

#ifdef __cplusplus
}
#endif

#endif // UCC_TL_CUDA_ALLREDUCE_KERNEL_H_
