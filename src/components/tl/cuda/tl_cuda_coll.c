/**
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_coll.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "bcast/bcast.h"
#include "reduce_scatter/reduce_scatter.h"
#include "reduce_scatterv/reduce_scatterv.h"
#include "allreduce/allreduce.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

#include <string.h>

#if ENABLE_DEBUG == 1
/* TODO: possible need to check CUDA context */
#define UCC_TL_CUDA_CHECK_DEVICE_MATCH(_team) do {                             \
    int _dev;                                                                  \
    CUDA_CHECK(cudaGetDevice(&_dev));                                          \
    if (_dev != UCC_TL_CUDA_TEAM_CTX(_team)->device) {                         \
        tl_error(UCC_TL_TEAM_LIB(_team), "CUDA device mismatch, "              \
                 "current device %d, team device %d\n", _dev,                  \
                 UCC_TL_CUDA_TEAM_CTX(_team)->device);                         \
        return UCC_ERR_INVALID_PARAM;                                          \
    }                                                                          \
} while(0)
#else
#define UCC_TL_CUDA_CHECK_DEVICE_MATCH(_team)
#endif

const char *
    ucc_tl_cuda_default_alg_select_str[UCC_TL_CUDA_N_DEFAULT_ALG_SELECT_STR] = {
        UCC_TL_CUDA_ALLGATHER_DEFAULT_ALG_SELECT_STR,
        UCC_TL_CUDA_ALLGATHERV_DEFAULT_ALG_SELECT_STR,
        UCC_TL_CUDA_ALLREDUCE_DEFAULT_ALG_SELECT_STR,
        UCC_TL_CUDA_BCAST_DEFAULT_ALG_SELECT_STR,
        UCC_TL_CUDA_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR,
        UCC_TL_CUDA_REDUCE_SCATTERV_DEFAULT_ALG_SELECT_STR};

ucc_status_t ucc_tl_cuda_mem_info_get(void *ptr, size_t length,
                                      ucc_tl_cuda_mem_info_t *mi)
{
    ucc_mem_attr_t mem_attr;
    ucc_status_t   status;

    if (ptr == NULL || length == 0) {
        mi->ptr    = NULL;
        mi->length = 0;
        mi->offset = 0;
        return UCC_OK;
    }

    mem_attr.field_mask =
        UCC_MEM_ATTR_FIELD_BASE_ADDRESS | UCC_MEM_ATTR_FIELD_ALLOC_LENGTH;
    mem_attr.alloc_length = length;
    status                = ucc_mc_get_mem_attr(ptr, &mem_attr);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    mi->ptr    = mem_attr.base_address;
    mi->length = mem_attr.alloc_length;
    mi->offset = (ptrdiff_t)ptr - (ptrdiff_t)mi->ptr;
    CUDA_CHECK_GOTO(cudaIpcGetMemHandle(&mi->handle, mi->ptr), exit, status);
exit:
    return status;
}

ucc_status_t ucc_tl_cuda_mem_info_from_memh(ucc_mem_map_mem_h       memh,
                                            ucc_tl_cuda_mem_info_t *mi)
{
    ucc_mem_map_memh_t      *map_memh = (ucc_mem_map_memh_t *)memh;
    ucc_tl_cuda_memh_data_t *cuda_data;
    int                      i;

    if (!map_memh || !map_memh->tl_h) {
        return UCC_ERR_INVALID_PARAM;
    }

    /* Search for CUDA TL handle in the memory map handle */
    for (i = 0; i < map_memh->num_tls; i++) {
        if (strncmp(
                map_memh->tl_h[i].tl_name, "cuda", UCC_MEM_MAP_TL_NAME_LEN) ==
            0) {
            cuda_data = (ucc_tl_cuda_memh_data_t *)map_memh->tl_h[i].tl_data;
            if (!cuda_data) {
                /* CUDA TL was registered but has no data (e.g., non-device memory) */
                return UCC_ERR_NOT_FOUND;
            }

            /* Found CUDA TL handle - extract the IPC handle */
            mi->ptr    = cuda_data->base_address;
            mi->length = cuda_data->length;
            mi->offset = cuda_data->offset;
            memcpy(
                &mi->handle,
                &cuda_data->ipc_handle,
                sizeof(cudaIpcMemHandle_t));

            return UCC_OK;
        }
    }

    /* No CUDA TL handle found in the memory map */
    return UCC_ERR_NOT_FOUND;
}

ucc_status_t ucc_tl_cuda_mem_info_from_global_memh(
    ucc_mem_map_mem_h *global_memh, ucc_rank_t peer_rank,
    ucc_tl_cuda_mem_info_t *mi)
{
    if (!global_memh || !mi) {
        return UCC_ERR_INVALID_PARAM;
    }
    return ucc_tl_cuda_mem_info_from_memh(global_memh[peer_rank], mi);
}

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task_h)
{
    UCC_TL_CUDA_CHECK_DEVICE_MATCH(ucc_derived_of(team, ucc_tl_cuda_team_t));
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_tl_cuda_alltoall_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_ALLGATHER:
        return ucc_tl_cuda_allgather_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_ALLGATHERV:
        return ucc_tl_cuda_allgatherv_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_cuda_bcast_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return ucc_tl_cuda_reduce_scatter_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return ucc_tl_cuda_reduce_scatterv_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_cuda_allreduce_init(coll_args, team, task_h);
    case UCC_COLL_TYPE_ALLTOALLV:
        return ucc_tl_cuda_alltoallv_init(coll_args, team, task_h);
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
}

ucc_status_t ucc_tl_cuda_shm_barrier_init_root(ucc_rank_t size, ucc_rank_t rank, ucc_rank_t root,
                                          ucc_tl_cuda_shm_barrier_t *barrier)
{
    if (rank == root) {
        barrier->size  = size;
        barrier->count = 0;
        barrier->sense = 0;
    }
    barrier->state[rank]       = UCC_OK;
    barrier->local_sense[rank] = 1;
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_shm_barrier_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier)
{
    if (rank == 0) {
        barrier->size  = size;
        barrier->count = 0;
        barrier->sense = 0;
    }
    barrier->state[rank]       = UCC_OK;
    barrier->local_sense[rank] = 1;
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_shm_barrier_start(ucc_rank_t                 rank,
                                           ucc_tl_cuda_shm_barrier_t *barrier)
{
    ucc_rank_t pos = ucc_atomic_fadd32(&barrier->count, 1);

    barrier->state[rank] = UCC_INPROGRESS;
    if (pos == barrier->size - 1) {
        barrier->count = 0;
        ucc_memory_cpu_store_fence();
        barrier->sense = barrier->local_sense[rank];
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_shm_barrier_test(ucc_rank_t                 rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier)
{
    if (barrier->sense != barrier->local_sense[rank]) {
        return UCC_INPROGRESS;
    }
    barrier->state[rank]       = UCC_OK;
    barrier->local_sense[rank] = 1 - barrier->local_sense[rank];
    return UCC_OK;
}

static inline int alg_id_from_str(ucc_coll_type_t coll_type, const char *str)
{
    switch (coll_type) {
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_tl_cuda_alltoall_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALLV:
        return ucc_tl_cuda_alltoallv_alg_from_str(str);
    case UCC_COLL_TYPE_ALLGATHER:
        return ucc_tl_cuda_allgather_alg_from_str(str);
    case UCC_COLL_TYPE_ALLGATHERV:
        return ucc_tl_cuda_allgatherv_alg_from_str(str);
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_cuda_bcast_alg_from_str(str);
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_cuda_allreduce_alg_from_str(str);
    default:
        break;
    }
    return -1;
}

ucc_status_t ucc_tl_cuda_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t          coll_type,
                                        ucc_memory_type_t        mem_type,
                                        ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    if (alg_id_str) {
        alg_id = alg_id_from_str(coll_type, alg_id_str);
    }

    if (mem_type != UCC_MEMORY_TYPE_CUDA) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    switch (coll_type) {
    case UCC_COLL_TYPE_ALLTOALL:
        switch (alg_id) {
        case UCC_TL_CUDA_ALLTOALL_ALG_CE:
            *init = ucc_tl_cuda_alltoall_init;
            break;
        case UCC_TL_CUDA_ALLTOALL_ALG_PUSH:
            *init = ucc_tl_cuda_alltoall_push_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        switch (alg_id) {
        case UCC_TL_CUDA_ALLTOALLV_ALG_CE:
            *init = ucc_tl_cuda_alltoallv_init;
            break;
        case UCC_TL_CUDA_ALLTOALLV_ALG_PUSH:
            *init = ucc_tl_cuda_alltoallv_push_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        switch (alg_id) {
        case UCC_TL_CUDA_ALLGATHER_ALG_AUTO:
            *init = ucc_tl_cuda_allgather_init;
            break;
        case UCC_TL_CUDA_ALLGATHER_ALG_RING:
            *init = ucc_tl_cuda_allgather_ring_init;
            break;
        case UCC_TL_CUDA_ALLGATHER_ALG_LINEAR:
            *init = ucc_tl_cuda_allgather_linear_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        switch (alg_id) {
        case UCC_TL_CUDA_ALLGATHERV_ALG_AUTO:
            *init = ucc_tl_cuda_allgatherv_init;
            break;
        case UCC_TL_CUDA_ALLGATHERV_ALG_RING:
            *init = ucc_tl_cuda_allgatherv_ring_init;
            break;
        case UCC_TL_CUDA_ALLGATHERV_ALG_LINEAR:
            *init = ucc_tl_cuda_allgatherv_linear_init;
            break;
#ifdef HAVE_NVLS
        case UCC_TL_CUDA_ALLGATHERV_ALG_NVLS:
            *init = ucc_tl_cuda_allgatherv_nvls_init;
            break;
#endif /* HAVE_NVLS */
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_BCAST:
        switch (alg_id) {
        case UCC_TL_CUDA_BCAST_ALG_LINEAR:
            *init = ucc_tl_cuda_bcast_linear_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        switch (alg_id) {
#ifdef HAVE_NVLS
        case UCC_TL_CUDA_ALLREDUCE_ALG_NVLS:
            *init = ucc_tl_cuda_allreduce_nvls_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
#else
        default:
            status = UCC_ERR_NOT_SUPPORTED;
            break;
#endif /* HAVE_NVLS */
        };
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        switch (alg_id) {
        case UCC_TL_CUDA_REDUCE_SCATTER_ALG_AUTO:
            *init = ucc_tl_cuda_reduce_scatter_init;
            break;
        case UCC_TL_CUDA_REDUCE_SCATTER_ALG_RING:
            *init = ucc_tl_cuda_reduce_scatter_ring_init;
            break;
        case UCC_TL_CUDA_REDUCE_SCATTER_ALG_LINEAR:
            *init = ucc_tl_cuda_reduce_scatter_linear_init;
            break;
#ifdef HAVE_NVLS
        case UCC_TL_CUDA_REDUCE_SCATTER_ALG_NVLS:
            *init = ucc_tl_cuda_reduce_scatter_nvls_init;
            break;
#endif /* HAVE_NVLS */
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        switch (alg_id) {
        case UCC_TL_CUDA_REDUCE_SCATTERV_ALG_AUTO:
            *init = ucc_tl_cuda_reduce_scatterv_init;
            break;
        case UCC_TL_CUDA_REDUCE_SCATTERV_ALG_RING:
            *init = ucc_tl_cuda_reduce_scatterv_ring_init;
            break;
        case UCC_TL_CUDA_REDUCE_SCATTERV_ALG_LINEAR:
            *init = ucc_tl_cuda_reduce_scatterv_linear_init;
            break;
#ifdef HAVE_NVLS
        case UCC_TL_CUDA_REDUCE_SCATTERV_ALG_NVLS:
            *init = ucc_tl_cuda_reduce_scatterv_nvls_init;
            break;
#endif /* HAVE_NVLS */
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
