/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "bcast.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_bcast_algs[UCC_TL_UCP_BCAST_ALG_LAST + 1] = {
        [UCC_TL_UCP_BCAST_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_BCAST_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "bcast over knomial tree with arbitrary radix "
                     "(optimized for latency)"},
        [UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL] =
            {.id   = UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL,
             .name = "sag_knomial",
             .desc = "recursive knomial scatter followed by knomial "
                     "allgather (optimized for BW)"},
        [UCC_TL_UCP_BCAST_ALG_TWO_TREE] =
            {.id   = UCC_TL_UCP_BCAST_ALG_TWO_TREE,
             .name = "two_tree",
             .desc = "bcast over double binary tree where a leaf in one tree "
                     "will be intermediate in other (optimized for latency)"},
        [UCC_TL_UCP_BCAST_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_bcast_init(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         team_size = (ucc_rank_t)task->subset.map.ep_num;

    task->bcast_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.bcast_kn_radix, team_size);

    task->super.post     = ucc_tl_ucp_bcast_knomial_start;
    task->super.progress = ucc_tl_ucp_bcast_knomial_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_bcast_knomial_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *team,
                                           ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task    = ucc_tl_ucp_init_task(coll_args, team);
    status  = ucc_tl_ucp_bcast_init(task);
    *task_h = &task->super;
    return status;
}
