/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_NVLS_H_
#define UCC_TL_CUDA_NVLS_H_

#include <cuda.h>
#include "ucc/api/ucc_status.h"

// Forward declaration to avoid circular dependency
struct ucc_tl_cuda_team;
struct ucc_base_context;

typedef struct {
    pid_t pid;
    int   handle;
} ucc_tl_cuda_nvls_share_data_t;

typedef enum {
    UCC_TL_CUDA_NVLS_STATE_INIT,
    UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES,
    UCC_TL_CUDA_NVLS_STATE_IMPORT_HANDLE,
    UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE,
} ucc_tl_cuda_nvls_state_t;

typedef struct ucc_tl_cuda_nvls {
    // Multicast handle
    CUmemGenericAllocationHandle   mc_handle;
    // Multicast memory handle
    CUmemGenericAllocationHandle   mc_memhandle;
    // Device pointer for multicast
    CUdeviceptr                    mc_va;
    // Device pointer for unicast
    CUdeviceptr                    uc_va;
    // Size of multicast memory
    size_t                         mc_size;
    // Offset of multicast memory
    size_t                         mc_offset;
    // Coll id for each task
    size_t                        *coll_ids;
    // Whether the team is multi-node
    int                            is_multinode;
    // Temporary buffer for allgather
    ucc_tl_cuda_nvls_share_data_t *share_data;
    // State variables for re-entrant initialization
    // POSIX file descriptor handle
    int                            export_handle;
    // Fabric handle for multi-node
    CUmemFabricHandle              fabric_handle;
    // Array of PIDs from all ranks
    pid_t                         *shared_pids;
    // Array of fabric handles
    CUmemFabricHandle             *shared_fabric_handles;
    // CUDA device ID
    int                            device;
    // Minimum granularity
    size_t                         minGran;
    // Granularity
    size_t                         gran;
} ucc_tl_cuda_nvls_t;

typedef struct ucc_tl_cuda_nvls_control {
    uint64_t arrival_counter;
} ucc_tl_cuda_nvls_control_t;

ucc_status_t ucc_tl_cuda_nvls_init(
    struct ucc_tl_cuda_team *team, struct ucc_base_context *tl_context);

ucc_status_t ucc_tl_cuda_nvls_destroy(struct ucc_tl_cuda_team *team);

#endif // UCC_TL_CUDA_NVLS_H_
