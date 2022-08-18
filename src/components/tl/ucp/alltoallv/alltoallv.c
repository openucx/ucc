/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *task);

ucc_base_coll_alg_info_t
    ucc_tl_ucp_alltoallv_algs[UCC_TL_UCP_ALLTOALLV_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE] =
            {.id   = UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE,
             .name = "pairwise",
             .desc = "O(N) pairwise exchange with adjustable number "
             "of outstanding sends/recvs"},
        [UCC_TL_UCP_ALLTOALLV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

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
