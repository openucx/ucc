/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_tag.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"
#include "bcast/bcast.h"

//NOLINTNEXTLINE subset unused
static ucc_rank_t ucc_tl_ucp_service_ring_get_send_block(ucc_subset_t *subset,
                                                         ucc_rank_t trank,
                                                         ucc_rank_t tsize,
                                                         int step)
{
    return (trank - step + tsize) % tsize;
}

//NOLINTNEXTLINE subset unused
static ucc_rank_t ucc_tl_ucp_service_ring_get_recv_block(ucc_subset_t *subset,
                                                         ucc_rank_t trank,
                                                         ucc_rank_t tsize,
                                                         int step)
{
    return (trank - step - 1 + tsize) % tsize;
}

static ucc_status_t ucc_tl_ucp_service_coll_start_executor(ucc_coll_task_t *task)
{
    ucc_ee_executor_params_t eparams;
    ucc_status_t status;

    eparams.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
    eparams.ee_type = UCC_EE_CPU_THREAD;

    status = ucc_ee_executor_init(&eparams, &task->executor);
    if (status != UCC_OK) {
        return status;
    }

    status = ucc_ee_executor_start(task->executor, NULL);
    if (status != UCC_OK) {
        ucc_ee_executor_finalize(task->executor);
        return status;
    }

    task->flags |= UCC_COLL_TASK_FLAG_EXECUTOR_STOP;

    return UCC_OK;
}

static ucc_status_t ucc_tl_ucp_service_coll_stop_executor(ucc_coll_task_t *task)
{
    ucc_status_t status, gl_status;

    gl_status = UCC_OK;
    status = ucc_ee_executor_stop(task->executor);
    if (status != UCC_OK) {
        gl_status = status;
    }

    status = ucc_ee_executor_finalize(task->executor);
    if (status != UCC_OK) {
        gl_status = status;
    }

    return gl_status;
}

ucc_status_t ucc_tl_ucp_service_allreduce(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, ucc_datatype_t dt,
                                          size_t count, ucc_reduction_op_t op,
                                          ucc_subset_t      subset,
                                          ucc_coll_task_t **task_p)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_get_task(tl_team);
    ucc_status_t       status;

    ucc_base_coll_args_t bargs   = {
        .args = {
            .mask         = 0,
            .coll_type    = UCC_COLL_TYPE_ALLREDUCE,
            .op           = op,
            .src.info = {
                .buffer   = sbuf,
                .count    = count,
                .datatype = dt,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .dst.info = {
                .buffer   = rbuf,
                .count    = count,
                .datatype = dt,
                .mem_type = UCC_MEMORY_TYPE_HOST
            }
        }
    };

    status = ucc_coll_task_init(&task->super, &bargs, team);
    if (status != UCC_OK) {
        goto free_task;
    }
    task->flags          = UCC_TL_UCP_TASK_FLAG_SUBSET;
    task->subset         = subset;
    task->tagged.tag     = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls        = UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.oob_npolls;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_knomial_finalize;

    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
    if (status != UCC_OK) {
        goto free_task;
    }

    status = ucc_tl_ucp_service_coll_start_executor(&task->super);
    if (status != UCC_OK) {
        goto free_task;
    }

    status = ucc_tl_ucp_allreduce_knomial_start(&task->super);
    if (status != UCC_OK) {
        goto finalize_coll;
    }

    *task_p = &task->super;
    return status;

finalize_coll:
    ucc_tl_ucp_allreduce_knomial_finalize(&task->super);
    ucc_tl_ucp_service_coll_stop_executor(&task->super);
free_task:
    ucc_tl_ucp_put_task(task);
    return status;
}

ucc_status_t ucc_tl_ucp_service_allgather(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, size_t msgsize,
                                          ucc_subset_t      subset,
                                          ucc_coll_task_t **task_p)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t   *task     = ucc_tl_ucp_get_task(tl_team);
    uint32_t             npolls   =
        UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.oob_npolls;
    int                  in_place =
        sbuf == PTR_OFFSET(rbuf, msgsize * subset.myrank);
    ucc_base_coll_args_t bargs    = {
        .args = {
            .coll_type = UCC_COLL_TYPE_ALLGATHER,
            .mask      = UCC_COLL_ARGS_FIELD_FLAGS,
            .flags     = in_place ? UCC_COLL_ARGS_FLAG_IN_PLACE : 0,
            .src.info = {.buffer   = sbuf,
                         .count    = msgsize,
                         .datatype = UCC_DT_UINT8,
                         .mem_type = UCC_MEMORY_TYPE_HOST},
            .dst.info = {.buffer   = rbuf,
                         .count    = msgsize * subset.map.ep_num,
                         .datatype = UCC_DT_UINT8,
                         .mem_type = UCC_MEMORY_TYPE_HOST}
        }
    };
    ucc_status_t       status;

    status               = ucc_coll_task_init(&task->super, &bargs, team);
    if (status != UCC_OK) {
        goto free_task;
    }
    task->allgather_ring.get_send_block = ucc_tl_ucp_service_ring_get_send_block;
    task->allgather_ring.get_recv_block = ucc_tl_ucp_service_ring_get_recv_block;
    task->flags                         = UCC_TL_UCP_TASK_FLAG_SUBSET;
    task->subset                        = subset;
    task->tagged.tag                    = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls                       = npolls;
    task->super.progress                = ucc_tl_ucp_allgather_ring_progress;
    task->super.finalize                = ucc_tl_ucp_coll_finalize;

    status = ucc_tl_ucp_allgather_ring_start(&task->super);
    if (status != UCC_OK) {
        goto finalize_coll;
    }

    *task_p = &task->super;
    return status;
finalize_coll:
    ucc_tl_ucp_coll_finalize(*task_p);
free_task:
    ucc_tl_ucp_put_task(task);
    return status;
}

ucc_status_t ucc_tl_ucp_service_bcast(ucc_base_team_t *team, void *buf,
                                      size_t msgsize, ucc_rank_t root,
                                      ucc_subset_t      subset,
                                      ucc_coll_task_t **task_p)
{
    ucc_tl_ucp_team_t   *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t   *task    = ucc_tl_ucp_get_task(tl_team);
    ucc_base_coll_args_t bargs   = {
        .args = {
            .coll_type    = UCC_COLL_TYPE_BCAST,
            .src.info = {
                .buffer   = buf,
                .count    = msgsize,
                .datatype = UCC_DT_INT8,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .root         = root
        }
    };
    ucc_status_t status;

    status = ucc_coll_task_init(&task->super, &bargs, team);
    if (status != UCC_OK) {
        goto free_task;
    }
    task->flags          = UCC_TL_UCP_TASK_FLAG_SUBSET;
    task->subset         = subset;
    task->tagged.tag     = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls        = UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.oob_npolls;
    task->super.progress = ucc_tl_ucp_bcast_knomial_progress;
    task->super.finalize = ucc_tl_ucp_coll_finalize;
    task->bcast_kn.radix = 2;
    status = ucc_tl_ucp_bcast_knomial_start(&task->super);
    if (status != UCC_OK) {
        goto finalize_coll;
    }

    *task_p = &task->super;
    return status;
finalize_coll:
    ucc_tl_ucp_coll_finalize(*task_p);
free_task:
    ucc_tl_ucp_put_task(task);
    return status;
}

void ucc_tl_ucp_service_update_id(ucc_base_team_t *team, uint16_t id) {
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);

    tl_team->super.super.params.id  = id;
}
