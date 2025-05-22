/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"

#define ALLTOALL_MAX_PATTERN_SIZE (sizeof(UCC_TL_UCP_ALLTOALL_DEFAULT_ALG_SELECT_STR_PATTERN) + 32)
#define ALLTOALL_DEFAULT_ALG_SWITCH 129

ucc_status_t ucc_tl_ucp_alltoall_pairwise_start(ucc_coll_task_t *task);
void ucc_tl_ucp_alltoall_pairwise_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *task);
void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *task);

char* ucc_tl_ucp_alltoall_score_str_get(ucc_tl_ucp_team_t *team)
{
    int max_size = ALLTOALL_MAX_PATTERN_SIZE;
    char *str;

    str = ucc_malloc(max_size * sizeof(char));
    ucc_snprintf_safe(str, max_size,
                      UCC_TL_UCP_ALLTOALL_DEFAULT_ALG_SELECT_STR_PATTERN,
                      ALLTOALL_DEFAULT_ALG_SWITCH * UCC_TL_TEAM_SIZE(team));
    return str;
}

ucc_base_coll_alg_info_t
    ucc_tl_ucp_alltoall_algs[UCC_TL_UCP_ALLTOALL_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE,
             .name = "pairwise",
             .desc = "pairwise two-sided implementation"},
        [UCC_TL_UCP_ALLTOALL_ALG_BRUCK] =
            {.id   = UCC_TL_UCP_ALLTOALL_ALG_BRUCK,
             .name = "bruck",
             .desc = "Bruck alltoall"},
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
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h)
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
