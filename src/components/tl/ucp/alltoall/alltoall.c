/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"

ucc_status_t ucc_tl_ucp_alltoall_pairwise_start(ucc_coll_task_t *task);
void ucc_tl_ucp_alltoall_pairwise_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *task);
void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *task);

ucc_base_coll_alg_info_t
    ucc_tl_ucp_alltoall_algs[UCC_TL_UCP_ALLTOALL_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE,
             .name = "pairwise",
             .desc = "pairwise two-sided implementation"},
        [UCC_TL_UCP_ALLTOALL_ALG_ONESIDED] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_ONESIDED,
             .name = "onesided",
             .desc = "naive, linear one-sided implementation"},
        [UCC_TL_UCP_ALLTOALL_ALG_LAST] = {.id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_alltoall_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;

    ALLTOALL_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    status = ucc_tl_ucp_alltoall_pairwise_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    status = ucc_tl_ucp_alltoall_pairwise_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_alltoall_onesided_start;
    task->super.progress = ucc_tl_ucp_alltoall_onesided_progress;
    status               = UCC_OK;
out:
    return status;
}
