/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "mcast/tl_mlx5_mcast_coll.h"
#include "alltoall/alltoall.h"

ucc_status_t ucc_tl_mlx5_bcast_mcast_init(ucc_base_coll_args_t *coll_args,
                                          ucc_base_team_t      *team,
                                          ucc_coll_task_t     **task_h)
{
    ucc_status_t        status = UCC_OK;
    ucc_tl_mlx5_task_t *task   = NULL;
    
    status = ucc_tl_mlx5_mcast_check_support(coll_args, team);
    if (UCC_OK != status) {
        return status;
    }

    task = ucc_tl_mlx5_get_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    task->super.finalize = ucc_tl_mlx5_task_finalize;

    status = ucc_tl_mlx5_mcast_bcast_init(task);
    if (ucc_unlikely(UCC_OK != status)) {
        goto free_task;
    }

    *task_h = &(task->super);

    tl_debug(UCC_TASK_LIB(task), "init coll task %p", task);

    return UCC_OK;

free_task:
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_mlx5_task_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_mcast_coll_req_t *req = task->bcast_mcast.req_handle;

    if (req != NULL) {
        ucc_assert(coll_task->status != UCC_INPROGRESS);
        ucc_free(req);
        tl_trace(UCC_TASK_LIB(task), "finalizing an mcast task %p", task);
        task->bcast_mcast.req_handle = NULL;
    }

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_mlx5_put_task(task);
    return UCC_OK;
}

ucc_tl_mlx5_task_t* ucc_tl_mlx5_init_task(ucc_base_coll_args_t *coll_args,
                                          ucc_base_team_t      *team,
                                          ucc_schedule_t       *schedule)
{
    ucc_tl_mlx5_task_t *task = ucc_tl_mlx5_get_task(coll_args, team);

    task->super.schedule = schedule;
    task->super.finalize = ucc_tl_mlx5_task_finalize;
    return task;
}

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task_h)
{
    ucc_status_t status = UCC_OK;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_mlx5_alltoall_init(coll_args, team, task_h);
        break;
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_mlx5_bcast_mcast_init(coll_args, team, task_h);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }

    return status;
}
