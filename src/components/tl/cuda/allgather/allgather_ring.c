/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "allgatherv/allgatherv.h"
#include "allgather/allgather.h"

//NOLINTNEXTLINE

ucc_status_t ucc_tl_cuda_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     tl_team,
                                             ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task = ucc_tl_cuda_task_init(coll_args, team);

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    task->allgatherv_ring.get_count  = ucc_tl_cuda_allgather_get_count;
    task->allgatherv_ring.get_offset = ucc_tl_cuda_allgather_get_offset;
    task->allgatherv_ring.dt         = coll_args->args.dst.info.datatype;
    task->allgatherv_ring.sbuf       = coll_args->args.src.info.buffer;
    task->allgatherv_ring.rbuf       = coll_args->args.dst.info.buffer;

    task->super.flags               |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post                = ucc_tl_cuda_allgatherv_ring_start;
    task->super.triggered_post      = ucc_triggered_post;
    task->super.progress            = ucc_tl_cuda_allgatherv_ring_progress;
    task->super.finalize            = ucc_tl_cuda_allgatherv_ring_finalize;
    task->bar                       = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
