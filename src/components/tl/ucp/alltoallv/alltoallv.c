/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"

/* TODO: add as parameters */
#define NP_THRESH 32

ucc_base_coll_alg_info_t
    ucc_tl_ucp_alltoallv_algs[UCC_TL_UCP_ALLTOALLV_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE] =
            {.id   = UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE,
             .name = "pairwise",
             .desc = "O(N) pairwise exchange with adjustable number "
                     "of outstanding sends/recvs"},
        [UCC_TL_UCP_ALLTOALLV_ALG_HYBRID] =
            {.id   = UCC_TL_UCP_ALLTOALLV_ALG_HYBRID,
             .name = "hybrid",
             .desc = "hybrid a2av alg "},
        [UCC_TL_UCP_ALLTOALLV_ALG_ONESIDED] =
            {.id   = UCC_TL_UCP_ALLTOALLV_ALG_ONESIDED,
             .name = "onesided",
             .desc = "O(N) onesided alltoallv"},
        [UCC_TL_UCP_ALLTOALLV_ALG_LINEAR] =
            {.id   = UCC_TL_UCP_ALLTOALLV_ALG_LINEAR,
             .name = "linear",
             .desc = "O(N) linear alltoallv"},
        [UCC_TL_UCP_ALLTOALLV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

static ucc_rank_t get_num_posts(const ucc_tl_ucp_team_t *team)
{
    unsigned long posts = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    ucc_rank_t    tsize = UCC_TL_TEAM_SIZE(team);

    if (posts == UCC_ULUNITS_AUTO) {
        if (UCC_TL_TEAM_SIZE(team) <= NP_THRESH) {
            /* use linear algorithm */
            posts = 0;
        } else {
            /* use pairwise algorithm */
            posts = 1;
        }
    }

    posts = (posts > tsize || posts == 0) ? tsize: posts;
    return posts;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init_num_posts(ucc_base_coll_args_t *coll_args,
                                                         ucc_base_team_t *team,
                                                         ucc_coll_task_t **task_h,
                                                         ucc_rank_t num_posts)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLTOALLV_TASK_CHECK(coll_args->args, tl_team);
    task = ucc_tl_ucp_init_task(coll_args, team);
    task->alltoallv_pairwise.num_posts = num_posts;
    *task_h = &task->super;
    status = ucc_tl_ucp_alltoallv_pairwise_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t num_posts;

    num_posts = get_num_posts(tl_team);
    return ucc_tl_ucp_alltoallv_pairwise_init_num_posts(coll_args, team, task_h,
                                                        num_posts);
}

ucc_status_t ucc_tl_ucp_alltoallv_linear_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_coll_task_t     **task_h)
{
    return ucc_tl_ucp_alltoallv_pairwise_init_num_posts(coll_args, team, task_h, 0);
}

ucc_status_t ucc_tl_ucp_alltoallv_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;

    ALLTOALLV_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    task->alltoallv_pairwise.num_posts = get_num_posts(TASK_TEAM(task));
    status = ucc_tl_ucp_alltoallv_pairwise_init_common(task);
out:
    return status;
}
