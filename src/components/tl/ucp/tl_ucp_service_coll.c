/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_tag.h"
#include "allreduce/allreduce.h"

ucc_status_t ucc_tl_ucp_service_allreduce(ucc_base_team_t *team, void *sbuf,
                                          void *rbuf, ucc_datatype_t dt,
                                          size_t count, ucc_reduction_op_t op,
                                          ucc_tl_team_subset_t subset,
                                          ucc_coll_task_t **task_p)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_get_task(tl_team);
    ucc_status_t status;
    ucc_coll_args_t args = {
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
    };
    status = ucc_coll_task_init(&task->super, &args, team);
    if (status != UCC_OK) {
        goto free_task;
    }
    task->subset = subset;
    task->tag  = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls = 10; // TODO need a var ?
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    *task_p = &task->super;
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
    if (status != UCC_OK) {
        goto free_task;
    }
    status = ucc_tl_ucp_allreduce_knomial_start(&task->super);
    if (status != UCC_OK) {
        goto finalize_coll;
    }

    return status;
finalize_coll:
   ucc_tl_ucp_allreduce_knomial_finalize(*task_p);
free_task:
    ucc_tl_ucp_put_task(task);
    return status;
}

ucc_status_t ucc_tl_ucp_service_test(ucc_coll_task_t *task)
{
    return task->super.status;
}


void ucc_tl_ucp_service_cleanup(ucc_coll_task_t *task)
{
    ucc_tl_ucp_allreduce_knomial_finalize(task);
}

void ucc_tl_ucp_service_update_id(ucc_base_team_t *team, uint16_t id) {
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    tl_team->id                = id;
}
