/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_alltoallv_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;

    ALLTOALLV_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    status = ucc_tl_ucp_alltoallv_pairwise_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLTOALLV_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    status = ucc_tl_ucp_alltoallv_pairwise_init_common(task);
out:
    return status;
}
