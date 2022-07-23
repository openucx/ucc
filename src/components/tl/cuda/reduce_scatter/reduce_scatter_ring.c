/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv/reduce_scatterv.h"

size_t ucc_tl_cuda_reduce_scatter_ring_count(const ucc_tl_cuda_task_t *task,
                                             ucc_rank_t block) //NOLINT
{
    const ucc_coll_args_t *args  = &TASK_ARGS(task);
    size_t                 count = args->dst.info.count;

    if (UCC_IS_INPLACE(*args)) {
        count = args->dst.info.count / UCC_TL_TEAM_SIZE(TASK_TEAM(task));
    }
    return count;
}

size_t ucc_tl_cuda_reduce_scatter_ring_get_offset(const ucc_tl_cuda_task_t *task,
                                                  ucc_rank_t block)
{
    return ucc_tl_cuda_reduce_scatter_ring_count(task, block) * block;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    size_t              ssize  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
    ucc_datatype_t      dt     = args->dst.info.datatype;
    size_t send_size, frag_size;

    task->reduce_scatterv_ring.get_count  = ucc_tl_cuda_reduce_scatter_ring_count;
    task->reduce_scatterv_ring.get_offset = ucc_tl_cuda_reduce_scatter_ring_get_offset;
    task->reduce_scatterv_ring.dt         = args->dst.info.datatype;
    task->reduce_scatterv_ring.sbuf       = args->src.info.buffer;
    task->reduce_scatterv_ring.rbuf       = args->dst.info.buffer;

    send_size = task->reduce_scatterv_ring.get_count(task, 0);
    frag_size = ucc_min(ssize / ucc_dt_size(dt) / 2, send_size);

    task->super.flags                    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->reduce_scatterv_ring.num_frags = ucc_div_round_up(send_size, frag_size);
    task->super.post                     = ucc_tl_cuda_reduce_scatterv_ring_start;
    task->super.triggered_post           = ucc_triggered_post;
    task->super.progress                 = ucc_tl_cuda_reduce_scatterv_ring_progress;
    task->super.finalize                 = ucc_tl_cuda_reduce_scatterv_ring_finalize;
    task->bar                            = TASK_BAR(task);

    return UCC_OK;
}
