/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv/allgatherv.h"
#include "allgather/allgather.h"

ucc_status_t ucc_tl_ucp_allgatherv_knomial_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *team,
                                                ucc_coll_task_t **task_h)
{
    if (!UCC_COLL_IS_DST_CONTIG(&coll_args->args)) {
        return ucc_tl_ucp_allgatherv_ring_init(coll_args, team, task_h);
    }
    return ucc_tl_ucp_allgather_knomial_init(coll_args, team, task_h);
}
