/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "allgatherv/allgatherv.h"
#include "allgather/allgather.h"


ucc_status_t ucc_tl_ucp_allgatherv_knomial_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *team,
                                                ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         tsize   = UCC_TL_TEAM_SIZE(tl_team);

    if (!ucc_coll_args_is_disp_contig(&coll_args->args, tsize)) {
        return ucc_tl_ucp_allgatherv_ring_init(coll_args, team, task_h);
    }

    return ucc_tl_ucp_allgather_knomial_init(coll_args, team, task_h);
}
