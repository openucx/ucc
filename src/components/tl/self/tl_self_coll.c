/**
 * Copyright (C) Mellanox Technologies Ltd. 2021-2022.  ALL RIGHTS RESERVED.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc.h"
#include "tl_self.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_malloc.h"

static inline ucc_tl_self_task_t *
ucc_tl_self_coll_init_task(ucc_base_coll_args_t *coll_args,
                           ucc_base_team_t      *team)
{
    ucc_tl_self_team_t    *tl_team = ucc_derived_of(team, ucc_tl_self_team_t);
    ucc_tl_self_context_t *ctx     = UCC_TL_SELF_TEAM_CTX(tl_team);
    ucc_tl_self_task_t    *task    = ucc_mpool_get(&ctx->req_mp);

    if (ucc_unlikely(!task)) {
        return NULL;
    }

    ucc_coll_task_init(&task->super, coll_args, team);
    UCC_TL_SELF_PROFILE_REQUEST_NEW(task, "tl_self_task", 0);
    task->super.finalize       = ucc_tl_self_coll_finalize;
    task->super.triggered_post = ucc_triggered_post;
    task->src = task->dst = NULL;
    task->size            = 0;
    return task;
}

static inline void ucc_tl_self_put_task(ucc_tl_self_task_t *task)
{
    UCC_TL_SELF_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

ucc_status_t ucc_tl_self_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_self_task_t *task = ucc_derived_of(coll_task, ucc_tl_self_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_self_put_task(task);
    return UCC_OK;
}

void ucc_tl_self_noop_progress(ucc_coll_task_t *task)
{
    task->status = UCC_OK;
}

void ucc_tl_self_coll_copy_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_self_task_t *task = ucc_derived_of(coll_task, ucc_tl_self_task_t);

    task->super.status = ucc_mc_memcpy(task->dst, task->src, task->size,
                                       task->dst_memtype, task->src_memtype);
}

ucc_status_t ucc_tl_self_coll_start(ucc_coll_task_t *task)
{
    return ucc_progress_queue_enqueue(UCC_TASK_CORE_CTX(task)->pq, task);
}

ucc_status_t ucc_tl_self_coll_noop_init(ucc_tl_self_task_t *task)
{
    task->super.post     = ucc_tl_self_coll_start;
    task->super.progress = ucc_tl_self_noop_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_self_coll_copy_init(ucc_tl_self_task_t *task)
{
    ucc_coll_args_t *args = &(task->super.bargs.args);

    task->super.post = ucc_tl_self_coll_start;
    if (UCC_IS_INPLACE(*args)) {
        /* no copy is required for in-place */
        task->super.progress = ucc_tl_self_noop_progress;
    } else {
        task->dst = args->dst.info.buffer;
        task->src = args->src.info.buffer;
        task->size =
            args->src.info.count * ucc_dt_size(args->src.info.datatype);
        task->dst_memtype    = args->dst.info.mem_type;
        task->src_memtype    = args->src.info.mem_type;
        task->super.progress = ucc_tl_self_coll_copy_progress;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_self_alltoallv_init(ucc_tl_self_task_t *task)
{
    ucc_coll_args_t *args = &(task->super.bargs.args);

    task->super.post = ucc_tl_self_coll_start;
    if (UCC_IS_INPLACE(*args)) {
        /* no copy is required for in-place */
        task->super.progress = ucc_tl_self_noop_progress;
    } else {
        size_t displ = (size_t)ucc_coll_args_get_displacement(
            args, args->dst.info_v.displacements, 0);
        task->dst = PTR_OFFSET(args->dst.info_v.buffer, displ);
        displ     = (size_t)ucc_coll_args_get_displacement(
                args, args->src.info_v.displacements, 0);
        task->src  = PTR_OFFSET(args->src.info_v.buffer, displ);
        task->size = ucc_coll_args_get_count(args, args->src.info_v.counts, 0) *
                     ucc_dt_size(args->src.info_v.datatype);
        task->dst_memtype    = args->dst.info_v.mem_type;
        task->src_memtype    = args->src.info_v.mem_type;
        task->super.progress = ucc_tl_self_coll_copy_progress;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_self_coll_copyv_init(ucc_tl_self_task_t *task)
{
    ucc_coll_args_t *args = &(task->super.bargs.args);

    task->super.post = ucc_tl_self_coll_start;
    if (UCC_IS_INPLACE(*args)) {
        /* no copy is required for in-place */
        task->super.progress = ucc_tl_self_noop_progress;
    } else {
        size_t displ = 0;
        /* reduce_scatterv may not provide displacements */
        if (args->dst.info_v.displacements) {
            displ = (size_t)ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, 0);
        }
        task->dst = PTR_OFFSET(args->dst.info_v.buffer, displ);
        task->src = args->src.info.buffer;
        task->size =
            args->src.info.count * ucc_dt_size(args->src.info.datatype);
        task->dst_memtype    = args->dst.info_v.mem_type;
        task->src_memtype    = args->src.info.mem_type;
        task->super.progress = ucc_tl_self_coll_copy_progress;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_self_scatterv_init(ucc_tl_self_task_t *task)
{
    ucc_coll_args_t *args = &(task->super.bargs.args);

    task->super.post = ucc_tl_self_coll_start;
    if (UCC_IS_INPLACE(*args)) {
        /* no copy is required for in-place */
        task->super.progress = ucc_tl_self_noop_progress;
    } else {
        size_t displ = (size_t)ucc_coll_args_get_displacement(
            args, args->src.info_v.displacements, 0);
        task->src = PTR_OFFSET(args->src.info_v.buffer, displ);
        task->dst = args->dst.info.buffer;
        task->size =
            args->dst.info.count * ucc_dt_size(args->dst.info.datatype);
        task->dst_memtype    = args->dst.info.mem_type;
        task->src_memtype    = args->src.info_v.mem_type;
        task->super.progress = ucc_tl_self_coll_copy_progress;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_self_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task_h)
{
    ucc_status_t        status;
    ucc_tl_self_task_t *task = ucc_tl_self_coll_init_task(coll_args, team);

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_BCAST:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        status = ucc_tl_self_coll_noop_init(task);
        break;
    case UCC_COLL_TYPE_REDUCE:
    case UCC_COLL_TYPE_GATHER:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        status = ucc_tl_self_coll_copy_init(task);
        break;
    case UCC_COLL_TYPE_GATHERV:
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        status = ucc_tl_self_coll_copyv_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_self_alltoallv_init(task);
        break;
    case UCC_COLL_TYPE_SCATTERV:
        status = ucc_tl_self_scatterv_init(task);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_self_put_task(task);
        return status;
    }
    tl_trace(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}
