/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"
#include "components/mc/ucc_mc.h"

ucc_status_t ucc_tl_cuda_alltoall_ce_init(ucc_tl_cuda_task_t *task);

ucc_status_t ucc_tl_cuda_alltoall_ce_start(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_alltoall_ce_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_alltoall_ce_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_alltoall_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t      *tl_team,
                                       ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (UCC_IS_INPLACE(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_tl_cuda_alltoall_ce_init(task);
    if (ucc_unlikely(status != UCC_OK)) {
        goto free_task;
    }

    *task_p = &task->super;
    return UCC_OK;

free_task:
    ucc_tl_cuda_task_put(task);
    return status;
}
