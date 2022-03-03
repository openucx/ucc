/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "allgather.h"
#include "components/mc/ucc_mc.h"

ucc_status_t ucc_tl_cuda_allgather_ring_init(ucc_tl_cuda_task_t *task);

ucc_status_t ucc_tl_cuda_allgather_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *tl_team,
                                        ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task = ucc_tl_cuda_task_init(coll_args, team);
    ucc_status_t status;

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_tl_cuda_allgather_ring_init(task);
    if (ucc_unlikely(status != UCC_OK)) {
        goto free_task;
    }

    *task_p = &task->super;
    return UCC_OK;

free_task:
    ucc_tl_cuda_task_put(task);
    return status;
}
