/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "utils/arch/cpu.h"

static ucc_status_t ucc_tl_nccl_nb_progress(ucc_tl_nccl_task_t *task) {
#if NCCL_USE_NON_BLOCKING
    ucc_tl_nccl_team_t *team = TASK_TEAM(task);
    ncclResult_t nccl_status, st;

    if (task->nccl_progress_st == UCC_INPROGRESS) {
        st = ncclCommGetAsyncError(team->nccl_comm, &nccl_status);
        if (st != ncclSuccess ||
            (nccl_status != ncclSuccess && nccl_status != ncclInProgress)) {
            tl_error(UCC_TL_TEAM_LIB(team), "NCCL error %d %s",
                     st != ncclSuccess ? st : nccl_status,
                     ncclGetErrorString(st != ncclSuccess ? st : nccl_status));
            return UCC_ERR_NO_MESSAGE;
        }
        if (nccl_status == ncclInProgress) {
            return UCC_INPROGRESS;
        }
    }
#endif
    return UCC_OK;
}

void ucc_tl_nccl_event_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

    status = ucc_tl_nccl_nb_progress(task);
    if (status != UCC_OK) {
        coll_task->status = status;
        return;
    }

    ucc_assert(task->completed != NULL);
    status = ucc_ec_event_test(task->completed, UCC_EE_CUDA_STREAM);
    coll_task->status = status;
#ifdef HAVE_PROFILING_TL_NCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_coll_done", 0);
    }
#endif
}

void ucc_tl_nccl_driver_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

    status = ucc_tl_nccl_nb_progress(task);
    if (status != UCC_OK) {
        coll_task->status = status;
        return;
    }

    coll_task->status = task->host_status;
#ifdef HAVE_PROFILING_TL_NCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_coll_done", 0);
    }
#endif
}

static void ucc_tl_nccl_req_mpool_obj_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_coll_task_destruct(obj);
}

static void ucc_tl_nccl_req_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                           void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;

    ucc_coll_task_construct(&req->super);
    req->super.progress   = ucc_tl_nccl_event_collective_progress;
    req->nccl_progress_st = UCC_OK;
}


static ucc_mpool_ops_t ucc_tl_nccl_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_nccl_req_mpool_obj_init,
    .obj_cleanup   = ucc_tl_nccl_req_mpool_obj_cleanup
};

static ucc_status_t ucc_tl_nccl_req_mapped_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    cudaError_t cu_st;

    cu_st = cudaHostAlloc((void**)chunk_p, *size_p, cudaHostAllocMapped);
    if (cu_st != cudaSuccess) {
        return UCC_ERR_NO_MEMORY;
    }

    return UCC_OK;
}

static void ucc_tl_nccl_req_mapped_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *chunk)
{
    cudaFreeHost(chunk);
}

static void ucc_tl_nccl_req_mapped_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                                  void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;
    cudaError_t st;
    st = cudaHostGetDevicePointer((void **)(&req->dev_status),
                                  (void *)&req->host_status, 0);
    if (st != cudaSuccess) {
        req->super.status = UCC_ERR_NO_MESSAGE;
    }
    ucc_coll_task_construct(&req->super);
    req->super.progress   = ucc_tl_nccl_driver_collective_progress;
    req->nccl_progress_st = UCC_OK;
}

static ucc_mpool_ops_t ucc_tl_nccl_req_mapped_mpool_ops = {
    .chunk_alloc   = ucc_tl_nccl_req_mapped_mpool_chunk_malloc,
    .chunk_release = ucc_tl_nccl_req_mapped_mpool_chunk_free,
    .obj_init      = ucc_tl_nccl_req_mapped_mpool_obj_init,
    .obj_cleanup   = ucc_tl_nccl_req_mpool_obj_cleanup
};

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_nccl_context_config_t *tl_nccl_config =
        ucc_derived_of(config, ucc_tl_nccl_context_config_t);
    int mem_ops_attr = 0;
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_nccl_config->super,
                              params->context);
    memcpy(&self->cfg, tl_nccl_config, sizeof(*tl_nccl_config));
    if (self->cfg.sync_type != UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
#if CUDA_VERSION < 12000
        CUresult cu_st;
        CUdevice cu_dev;
        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st == CUDA_SUCCESS) {
            cu_st = cuDeviceGetAttribute(&mem_ops_attr,
                                        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                                        cu_dev);
        } else {
            tl_debug(self->super.super.lib, "failed to get cuda device");
        }
#else
        mem_ops_attr = 1;
#endif
        if (mem_ops_attr == 0) {
            if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
                tl_error(self->super.super.lib, "memops not supported");
                return UCC_ERR_NOT_SUPPORTED;
            }
            tl_debug(self->super.super.lib, "fallback to event completion sync");
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT;
        } else {
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS;
        }
    }
    ucc_assert(self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS ||
               self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT);
    if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
        tl_debug(self->super.super.lib, "using memops completion sync");
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mapped_mpool_ops,
                                params->thread_mode, "tl_nccl_req_mp");
    } else {
        tl_debug(self->super.super.lib, "using event completion sync");
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mpool_ops, params->thread_mode,
                                "tl_nccl_req_mp");
    }

    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_nccl_req mpool");
        return status;
    }
    // scratch buffer for barrier
    cudaError_t cuda_st = cudaMalloc(&self->scratch_buf, sizeof(float));
    if (cuda_st != cudaSuccess) {
        return UCC_ERR_NO_MEMORY;
    }
    tl_debug(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_context_t)
{
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
    cudaFree(self->scratch_buf);
    self->scratch_buf = NULL;
}

UCC_CLASS_DEFINE(ucc_tl_nccl_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_nccl_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_mem_map(const ucc_base_context_t *context, int type, /* NOLINT */
                                 void *memh, void *tl_h) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_nccl_mem_unmap(const ucc_base_context_t *context, int type, /* NOLINT */
                                   void *memh) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_nccl_memh_pack(const ucc_base_context_t *context, /* NOLINT */
                                   int type, void *memh, void **pack_buffer) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}
