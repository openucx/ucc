/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast/bcast.h"

enum
{
    STAGE_SYNC,    /*< Wait for free SYNC segment */
    STAGE_SETUP,   /*< Wait for memhandle setup to finish */
    STAGE_COPIES,  /*< Linear algorithm is running */
    STAGE_BARRIER, /*< Linear algorithm is done, waiting for
                    *  other ranks to finish */
};

ucc_status_t ucc_tl_cuda_bcast_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

void ucc_tl_cuda_bcast_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_status_t        st;
    (void) team;
    (void) st;
    task->super.status = UCC_INPROGRESS;
}

ucc_status_t ucc_tl_cuda_bcast_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *   args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt    = task->allgatherv_linear.dt;

    (void) tsize;
    (void) args;
    (void) dt;
    task->bcast_linear.stage         = STAGE_SYNC;
    // task->bcast_linear.sbuf          = args->src.info.buffer;


    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_bcast_linear_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     tl_team,
                                               ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_conntected(team->topo) ||
        UCC_TL_TEAM_SIZE(team) - 1 > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    // task->allgatherv_linear.get_count  = ucc_tl_cuda_allgather_get_count;
    // task->allgatherv_linear.get_offset = ucc_tl_cuda_allgather_get_offset;
    // task->allgatherv_linear.dt         = coll_args->args.dst.info.datatype;
    // task->allgatherv_linear.sbuf       = coll_args->args.src.info.buffer;
    // task->allgatherv_linear.rbuf       = coll_args->args.dst.info.buffer;

    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_bcast_linear_start;
    task->super.progress       = ucc_tl_cuda_bcast_linear_progress;
    task->super.finalize       = ucc_tl_cuda_bcast_linear_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}

