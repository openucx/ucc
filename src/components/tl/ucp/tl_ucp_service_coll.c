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
    ucc_coll_task_init(&task->super);
    task->subset = subset;
    task->team = tl_team;
    task->tag  = UCC_TL_UCP_SERVICE_TAG;
    task->n_polls = 10; // TODO need a var ?
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    memcpy(&task->args, &args, sizeof(ucc_coll_args_t));
    *task_p = &task->super;
    return ucc_tl_ucp_allreduce_knomial_start(&task->super);
}

ucc_status_t ucc_tl_ucp_service_test(ucc_coll_task_t *task)
{
    return task->super.status;
}


void ucc_tl_ucp_service_cleanup(ucc_coll_task_t *task)
{
    ucc_tl_ucp_task_t *tl_task = ucc_derived_of(task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_put_task(tl_task);
}
