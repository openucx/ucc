/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv/allgatherv.h"
#include "allgather/allgather.h"

ucc_status_t ucc_tl_cuda_allgather_linear_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     tl_team,
                                               ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task = ucc_tl_cuda_task_init(coll_args, team);

    task->allgatherv_linear.get_count  = ucc_tl_cuda_allgather_get_count;
    task->allgatherv_linear.get_offset = ucc_tl_cuda_allgather_get_offset;
    task->allgatherv_linear.dt         = coll_args->args.dst.info.datatype;
    task->allgatherv_linear.sbuf       = coll_args->args.src.info.buffer;
    task->allgatherv_linear.rbuf       = coll_args->args.dst.info.buffer;

    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_allgatherv_linear_start;
    task->super.triggered_post = ucc_triggered_post;
    task->super.progress       = ucc_tl_cuda_allgatherv_linear_progress;
    task->super.finalize       = ucc_tl_cuda_allgatherv_linear_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
