/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv/reduce_scatterv.h"
#include "reduce_scatter/reduce_scatter.h"

ucc_status_t ucc_tl_cuda_reduce_scatter_linear_init(ucc_base_coll_args_t *coll_args,
                                                    ucc_base_team_t *     tl_team,
                                                    ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team   = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    size_t              ssize  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
    ucc_datatype_t      dt     = coll_args->args.dst.info.datatype;
    size_t send_size, frag_size;
    ucc_tl_cuda_task_t *task;
    ucc_status_t status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->reduce_scatterv_linear.get_count  = ucc_tl_cuda_reduce_scatter_get_count;
    task->reduce_scatterv_linear.get_offset = ucc_tl_cuda_reduce_scatter_get_offset;
    task->reduce_scatterv_linear.dt         = coll_args->args.dst.info.datatype;
    task->reduce_scatterv_linear.rbuf       = coll_args->args.dst.info.buffer;

    send_size = task->reduce_scatterv_linear.get_count(task, 0);
    frag_size = ucc_min(ssize / ucc_dt_size(dt) / 2, send_size);

    task->reduce_scatterv_linear.num_frags = ucc_div_round_up(send_size,
                                                             frag_size);
    task->super.flags          |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_reduce_scatterv_linear_start;
    task->super.triggered_post = ucc_triggered_post;
    task->super.progress       = ucc_tl_cuda_reduce_scatterv_linear_progress;
    task->super.finalize       = ucc_tl_cuda_reduce_scatterv_linear_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
