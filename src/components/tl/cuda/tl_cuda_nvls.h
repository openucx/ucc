/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_NVLS_H_
#define UCC_TL_CUDA_NVLS_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>                   // For CU_MEM_CREATE_USAGE_MULTICAST
#include "components/base/ucc_base_iface.h" // For ucc_base_context_t

// Forward declaration to avoid circular dependency
struct ucc_tl_cuda_team;

// clang-format off
typedef struct ucc_tl_cuda_nvls {
    CUmemGenericAllocationHandle mc_handle;    // Multicast handle
    CUmemGenericAllocationHandle mc_memhandle; // Multicast memory handle
    CUdeviceptr                  mc_va;        // Device pointer for multicast memory
    CUdeviceptr                  uc_va;        // Device pointer for unicast memory
    size_t                       mc_size;      // Size of multicast memory
    size_t                       mc_offset;    // Offset of multicast memory
    size_t                      *coll_ids;     // Coll id for each task in flight slot
    int                          is_multinode; // Whether the team is multi-node
} ucc_tl_cuda_nvls_t;
// clang-format on

typedef struct ucc_tl_cuda_nvls_control {
    uint64_t arrival_counter;
} ucc_tl_cuda_nvls_control_t;

ucc_status_t ucc_tl_cuda_nvls_init(struct ucc_tl_cuda_team *self,
                                   ucc_base_context_t      *tl_context);

ucc_status_t ucc_tl_cuda_nvls_destroy(struct ucc_tl_cuda_team *self,
                                      ucc_base_context_t      *tl_context);

#endif // UCC_TL_CUDA_NVLS_H_
