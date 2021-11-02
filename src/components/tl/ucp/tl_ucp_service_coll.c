/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_tag.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"

ucc_status_t ucc_tl_ucp_service_allreduce(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, ucc_datatype_t dt,
                                          size_t count, ucc_reduction_op_t op,
                                          ucc_subset_t subset,
                                          ucc_coll_task_t **task_p)
{
    ucc_tl_ucp_team_t   *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t   *task    = ucc_tl_ucp_get_task(tl_team);
    ucc_base_coll_args_t bargs   = {
        .args = {
            .coll_type            = UCC_COLL_TYPE_ALLREDUCE,
            .mask                 = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS,
            .reduce.predefined_op = op,
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
    ucc_status_t status;

    status = ucc_coll_task_init(&task->super, &bargs, team);
    if (status != UCC_OK) {
        goto free_task;
    }
    task->subset = subset;
    task->tag  = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls        = UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.oob_npolls;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_knomial_finalize;

    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
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
   ucc_tl_ucp_allreduce_knomial_finalize(*task_p);
free_task:
    ucc_tl_ucp_put_task(task);
    return status;
}

ucc_status_t ucc_tl_ucp_service_allgather(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, size_t msgsize,
                                          ucc_subset_t subset,
                                          ucc_coll_task_t **task_p)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t   *task     = ucc_tl_ucp_get_task(tl_team);
    int                  in_place =
        (sbuf == PTR_OFFSET(rbuf, msgsize * ucc_ep_map_eval(subset.map,
                                                            subset.myrank)));
    ucc_base_coll_args_t bargs    = {
        .args = {
            .coll_type = UCC_COLL_TYPE_ALLGATHER,
            .mask      = in_place ? UCC_COLL_ARGS_FLAG_IN_PLACE : 0,
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
    task->subset         = subset;
    task->tag            = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls        = UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.oob_npolls;
    task->super.progress = ucc_tl_ucp_allgather_ring_progress;
    task->super.finalize = ucc_tl_ucp_coll_finalize;

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

void ucc_tl_ucp_service_update_id(ucc_base_team_t *team, uint16_t id) {
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);

    tl_team->super.super.params.id  = id;
}
