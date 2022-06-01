/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#include "allgatherv/allgatherv.h"

//NOLINTNEXTLINE
size_t ucc_tl_cuda_allgather_get_count(const ucc_tl_cuda_task_t *task,
                                       ucc_rank_t block)
{
    return TASK_ARGS(task).dst.info.count / UCC_TL_TEAM_SIZE(TASK_TEAM(task));
}

size_t ucc_tl_cuda_allgather_get_offset(const ucc_tl_cuda_task_t *task,
                                        ucc_rank_t block)
{
    return (TASK_ARGS(task).dst.info.count / UCC_TL_TEAM_SIZE(TASK_TEAM(task))) *
            block;
}

ucc_status_t ucc_tl_cuda_allgather_ring_init(ucc_tl_cuda_task_t *task)
{
    ucc_coll_args_t *args  = &TASK_ARGS(task);

    task->allgatherv_ring.get_count  = ucc_tl_cuda_allgather_get_count;
    task->allgatherv_ring.get_offset = ucc_tl_cuda_allgather_get_offset;
    task->allgatherv_ring.dt         = args->dst.info.datatype;
    task->allgatherv_ring.sbuf       = args->src.info.buffer;
    task->allgatherv_ring.rbuf       = args->dst.info.buffer;

    task->super.flags               |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post                = ucc_tl_cuda_allgatherv_ring_start;
    task->super.triggered_post      = ucc_triggered_post;
    task->super.progress            = ucc_tl_cuda_allgatherv_ring_progress;
    task->super.finalize            = ucc_tl_cuda_allgatherv_ring_finalize;
    task->bar                       = TASK_BAR(task);

    return UCC_OK;
}
