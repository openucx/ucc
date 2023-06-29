/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "mcast/tl_mlx5_mcast_coll.h"

static ucc_status_t ucc_tl_mlx5_task_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t *task = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    tl_debug(UCC_TASK_LIB(task), "finalizing coll task %p", task);
    UCC_TL_MLX5_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_bcast_mcast_init(ucc_base_coll_args_t *coll_args,
                                          ucc_base_team_t *     team,
                                          ucc_coll_task_t **    task_h)
{
    ucc_status_t        status = UCC_OK;
    ucc_tl_mlx5_task_t  *task = NULL;

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
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
