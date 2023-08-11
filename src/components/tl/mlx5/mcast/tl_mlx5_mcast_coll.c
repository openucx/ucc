/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"

ucc_status_t ucc_tl_mlx5_mcast_test(ucc_tl_mlx5_mcast_coll_req_t* req /* NOLINT */)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t mcast_coll_do_bcast(void* buf, int size, int root, void *mr, /* NOLINT */
                                 mcast_coll_comm_t *comm, /* NOLINT */
                                 int is_blocking, /* NOLINT */
                                 ucc_tl_mlx5_mcast_coll_req_t **task_req_handle /* NOLINT */)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_mlx5_mcast_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t       *task      = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_team_t       *mlx5_team = TASK_TEAM(task);
    ucc_tl_mlx5_mcast_team_t *team      = mlx5_team->mcast;
    ucc_coll_args_t          *args      = &TASK_ARGS_MCAST(task);
    ucc_datatype_t            dt        = args->src.info.datatype;
    size_t                    count     = args->src.info.count;
    ucc_rank_t                root      = args->root;
    ucc_status_t              status    = UCC_OK;
    size_t                    data_size = ucc_dt_size(dt) * count;
    void                     *buf       = args->src.info.buffer;
    mcast_coll_comm_t        *comm      = team->mcast_comm;

    task->bcast_mcast.req_handle = NULL;

    status = mcast_coll_do_bcast(buf, data_size, root, NULL, comm,
                            UCC_TL_MLX5_MCAST_ENABLE_BLOCKING, &task->bcast_mcast.req_handle);
    if (UCC_OK != status && UCC_INPROGRESS != status) {
        tl_error(UCC_TASK_LIB(task), "mcast_coll_do_bcast failed:%d", status);
        coll_task->status = status;
        return ucc_task_complete(coll_task);
    }

    coll_task->status = status;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(mlx5_team)->pq, &task->super);
}

void ucc_tl_mlx5_mcast_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task      = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_status_t                  status    = UCC_OK;
    ucc_tl_mlx5_mcast_coll_req_t *req       = task->bcast_mcast.req_handle;

    if (req != NULL) {
        status = ucc_tl_mlx5_mcast_test(req);
        if (UCC_OK == status) {
            coll_task->status = UCC_OK;
            ucc_free(req);
            task->bcast_mcast.req_handle  = NULL;
        }
    }
}

ucc_status_t ucc_tl_mlx5_mcast_bcast_init(ucc_tl_mlx5_task_t *task)
{

    task->super.post     = ucc_tl_mlx5_mcast_bcast_start;
    task->super.progress = ucc_tl_mlx5_mcast_collective_progress;

    return UCC_OK;
}
