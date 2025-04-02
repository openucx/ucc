/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "mcast/tl_mlx5_mcast_coll.h"
#include "mcast/tl_mlx5_mcast_allgather.h"
#include "mcast/tl_mlx5_mcast_rcache.h"
#include "alltoall/alltoall.h"

ucc_status_t ucc_tl_mlx5_coll_mcast_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t      *team,
                                         ucc_coll_task_t     **task_h)
{
    ucc_status_t        status  = UCC_OK;
    ucc_tl_mlx5_task_t *task    = NULL;
    
    status = ucc_tl_mlx5_mcast_check_support(coll_args, team);
    if (UCC_OK != status) {
        return status;
    }

    task = ucc_tl_mlx5_get_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    task->super.finalize = ucc_tl_mlx5_task_finalize;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_mlx5_mcast_bcast_init(task);
        if (ucc_unlikely(UCC_OK != status)) {
            goto free_task;
        }
        *task_h = &(task->super);
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_mlx5_mcast_allgather_init(task);
        if (ucc_unlikely(UCC_OK != status)) {
            goto free_task;
        }
        *task_h = &(task->super);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        tl_trace(team->context->lib, "mcast not supported for this collective type");
        goto free_task;
    }

    tl_trace(UCC_TASK_LIB(task), "initialized mcast collective task %p", task);

    return UCC_OK;

free_task:
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_mlx5_task_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_mcast_coll_req_t *req  = task->coll_mcast.req_handle;

    if (req != NULL) {
        ucc_assert(coll_task->status != UCC_INPROGRESS);
        ucc_assert(req->comm->ctx != NULL);
        if (coll_task->status == UCC_OK) {
            if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_ALLGATHER) {
                req->comm->allgather_comm.under_progress_counter++;
            } else {
                ucc_assert(coll_task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST);
                req->comm->bcast_comm.under_progress_counter++;
            }
             /* reset the reliability structures so that it gets initialized again for the next
            * allgather */
            req->comm->one_sided.reliability_ready = 0;
            req->comm->stalled                     = 0;
            req->comm->timer                       = 0;
        }
        if (req->rreg != NULL) {
            ucc_tl_mlx5_mcast_mem_deregister(req->comm->ctx, req->rreg);
            req->rreg = NULL;
        }
        if (req->recv_rreg != NULL) {
            ucc_tl_mlx5_mcast_mem_deregister(req->comm->ctx, req->recv_rreg);
            req->recv_rreg = NULL;
        }
        if (req->ag_schedule) {
            ucc_free(req->ag_schedule);
            req->ag_schedule = NULL;
        }
        ucc_mpool_put(req);
        tl_trace(UCC_TASK_LIB(task), "finalizing an mcast task %p", task);
        task->coll_mcast.req_handle = NULL;
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
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_mlx5_coll_mcast_init(coll_args, team, task_h);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }

    return status;
}
