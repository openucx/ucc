/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "gather.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_gather_algs[UCC_TL_UCP_GATHER_ALG_LAST + 1] = {
        [UCC_TL_UCP_GATHER_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_GATHER_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "gather over knomial tree with arbitrary radix "
                     "(optimized for latency)"},
        [UCC_TL_UCP_GATHER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_gather_init(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team    = TASK_TEAM(task);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(team);
    ucc_kn_radix_t radix;

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.gather_kn_radix, size);

    return ucc_tl_ucp_gather_knomial_init_common(task, radix);
}
