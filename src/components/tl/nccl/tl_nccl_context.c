/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "utils/arch/cpu.h"

void ucc_tl_nccl_event_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

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

    coll_task->status = task->host_status;
#ifdef HAVE_PROFILING_TL_NCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_coll_done", 0);
    }
#endif
}

static void ucc_tl_nccl_req_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                           void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;
    req->super.progress = ucc_tl_nccl_event_collective_progress;
}


static ucc_mpool_ops_t ucc_tl_nccl_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_nccl_req_mpool_obj_init,
    .obj_cleanup   = NULL
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
    req->super.progress = ucc_tl_nccl_driver_collective_progress;
}

static ucc_mpool_ops_t ucc_tl_nccl_req_mapped_mpool_ops = {
    .chunk_alloc   = ucc_tl_nccl_req_mapped_mpool_chunk_malloc,
    .chunk_release = ucc_tl_nccl_req_mapped_mpool_chunk_free,
    .obj_init      = ucc_tl_nccl_req_mapped_mpool_obj_init,
    .obj_cleanup   = NULL
};

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_nccl_context_config_t *tl_nccl_config =
        ucc_derived_of(config, ucc_tl_nccl_context_config_t);
    int mem_ops_attr = 0;
    ucc_status_t status;
    CUresult cu_st;
    CUdevice cu_dev;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_nccl_config->super,
                              params->context);
    memcpy(&self->cfg, tl_nccl_config, sizeof(*tl_nccl_config));
    if (self->cfg.sync_type != UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st == CUDA_SUCCESS) {
            cu_st = cuDeviceGetAttribute(&mem_ops_attr,
                                        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                                        cu_dev);
        } else {
            tl_info(self->super.super.lib, "failed to get cuda device");
        }
        if (mem_ops_attr == 0) {
            if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
                tl_error(self->super.super.lib, "memops not supported");
                return UCC_ERR_NOT_SUPPORTED;
            }
            tl_info(self->super.super.lib, "fallback to event completion sync");
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT;
        } else {
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS;
        }
    }
    ucc_assert(self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS ||
               self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT);
    if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
        tl_info(self->super.super.lib, "using memops completion sync");
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mapped_mpool_ops,
                                params->thread_mode, "tl_nccl_req_mp");
    } else {
        tl_info(self->super.super.lib, "using event completion sync");
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
    tl_info(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
    cudaFree(self->scratch_buf);
    self->scratch_buf = NULL;
}

UCC_CLASS_DEFINE(ucc_tl_nccl_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_nccl_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }
    attr->topo_required = 0;
    return UCC_OK;
}
