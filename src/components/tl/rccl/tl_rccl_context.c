/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_rccl.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "utils/arch/cpu.h"

void ucc_tl_rccl_event_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_rccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_rccl_task_t);
    ucc_status_t status;

    ucc_assert(task->completed != NULL);
    status = ucc_ec_event_test(task->completed, UCC_EE_ROCM_STREAM);
    coll_task->status = status;
#ifdef HAVE_PROFILING_TL_RCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_RCCL_PROFILE_REQUEST_EVENT(coll_task, "rccl_coll_done", 0);
    }
#endif
}

void ucc_tl_rccl_driver_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_rccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_rccl_task_t);

    coll_task->status = task->host_status;
#ifdef HAVE_PROFILING_TL_RCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_RCCL_PROFILE_REQUEST_EVENT(coll_task, "rccl_coll_done", 0);
    }
#endif
}

static void ucc_tl_rccl_req_mpool_obj_init(ucc_mpool_t *mp, void *obj, //NOLINT: mp is unused
                                           void *chunk) //NOLINT: chunk is unused
{
    ucc_tl_rccl_task_t *req = (ucc_tl_rccl_task_t*) obj;

    ucc_coll_task_construct(&req->super);
    req->super.progress = ucc_tl_rccl_event_collective_progress;
}

static void ucc_tl_rccl_req_mpool_obj_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT: mp is unused
{
    ucc_coll_task_destruct(obj);
}


static ucc_mpool_ops_t ucc_tl_rccl_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_rccl_req_mpool_obj_init,
    .obj_cleanup   = ucc_tl_rccl_req_mpool_obj_cleanup
};

UCC_CLASS_INIT_FUNC(ucc_tl_rccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_rccl_context_config_t *tl_rccl_config =
	ucc_derived_of(config, ucc_tl_rccl_context_config_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_rccl_config->super,
			      params->context);
    memcpy(&self->cfg, tl_rccl_config, sizeof(*tl_rccl_config));

    // RCCL does not currently support memops for synchronization
    if (self->cfg.sync_type == UCC_TL_RCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
        tl_error(self->super.super.lib, "memops not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (self->cfg.sync_type != UCC_TL_RCCL_COMPLETION_SYNC_TYPE_EVENT) {
        tl_debug(self->super.super.lib, "fallback to event completion sync");
        self->cfg.sync_type = UCC_TL_RCCL_COMPLETION_SYNC_TYPE_EVENT;
    }

    ucc_assert(self->cfg.sync_type == UCC_TL_RCCL_COMPLETION_SYNC_TYPE_EVENT);
    tl_debug(self->super.super.lib, "using event completion sync");
    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_rccl_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_tl_rccl_req_mpool_ops, params->thread_mode,
                            "tl_rccl_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_rccl_req mpool");
        return status;
    }
    // scratch buffer for barrier
    hipError_t hip_st = hipMalloc(&self->scratch_buf, sizeof(float));
    if (hip_st != hipSuccess) {
        return UCC_ERR_NO_MEMORY;
    }
    tl_debug(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_rccl_context_t)
{
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
    hipFree(self->scratch_buf);
    self->scratch_buf = NULL;
}

UCC_CLASS_DEFINE(ucc_tl_rccl_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_rccl_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
    return UCC_OK;
}

ucc_status_t ucc_tl_rccl_mem_map(const ucc_base_context_t *context, int type, /* NOLINT */
                                 void *memh, void *tl_h) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_rccl_mem_unmap(const ucc_base_context_t *context, int type, /* NOLINT */
                                   void *memh) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_rccl_memh_pack(const ucc_base_context_t *context, /* NOLINT */
                                   int type, void *memh, void **pack_buffer) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}
