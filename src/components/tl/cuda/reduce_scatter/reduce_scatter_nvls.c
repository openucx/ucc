/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

// #include "reduce_scatterv/reduce_scatterv.h"
#include "reduce_scatter/reduce_scatter.h"
#include <ucc/api/ucc.h>

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *   args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    // ucc_datatype_t      dt    = task->reduce_scatterv_nvls.dt;
    ucc_rank_t          i;
    size_t              send_size;

    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        task->reduce_scatterv_nvls.rbuf = args->dst.info.buffer;
    } else {
        task->reduce_scatterv_nvls.rbuf = args->dst.info_v.buffer;
    }

    // task->reduce_scatterv_nvls.stage = STAGE_SYNC;
    task->reduce_scatterv_nvls.sbuf  = args->src.info.buffer;
    send_size = task->reduce_scatterv_nvls.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size =
            ucc_max(send_size, task->reduce_scatterv_nvls.get_count(task, i));
    }

    if (send_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }

    // ssize = get_scratch_size(team, dt);
    // frag_size = ucc_min(ssize / 2 / ucc_dt_size(dt) / tsize, send_size);
    // task->reduce_scatterv_nvls.num_frags = ucc_div_round_up(send_size, frag_size);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_reduce_scatterv_nvls_progress(ucc_coll_task_t *task)
{
    task->status = UCC_OK;
    return;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_nvls_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *     tl_team,
                                                  ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
        UCC_TL_TEAM_SIZE(team) - 1 > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->reduce_scatterv_linear.get_count  =
        ucc_tl_cuda_reduce_scatter_get_count;
    task->reduce_scatterv_linear.get_offset =
        ucc_tl_cuda_reduce_scatter_get_offset;
    task->reduce_scatterv_linear.dt         = coll_args->args.dst.info.datatype;
    task->super.flags          |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_reduce_scatterv_nvls_start;
    task->super.progress       = ucc_tl_cuda_reduce_scatterv_nvls_progress;
    task->super.finalize       = ucc_tl_cuda_reduce_scatterv_nvls_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
