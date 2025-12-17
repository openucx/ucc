/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv/reduce_scatterv.h"
#include "reduce_scatter/reduce_scatter.h"

ucc_status_t ucc_tl_cuda_reduce_scatter_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team  = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_datatype_t      dt    = coll_args->args.dst.info.datatype;
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;
    size_t              offset_elements;
    size_t              count_elements;

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    offset_elements = ucc_tl_cuda_reduce_scatter_get_offset(task, trank);
    count_elements  = ucc_tl_cuda_reduce_scatter_get_count(task, trank);

    status          = ucc_tl_cuda_reduce_scatterv_nvls_init_common(
        task, dt, offset_elements, count_elements);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_cuda_task_put(task);
        return status;
    }

    *task_p = &task->super;
    return UCC_OK;
}
