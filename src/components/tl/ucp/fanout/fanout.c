/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "fanout.h"
#include "../bcast/bcast.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_fanout_algs[UCC_TL_UCP_FANOUT_ALG_LAST + 1] = {
        [UCC_TL_UCP_FANOUT_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_FANOUT_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "fanout over knomial tree with arbitrary radix"},
        [UCC_TL_UCP_FANOUT_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_fanout_init(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         team_size = UCC_TL_TEAM_SIZE(team);
    ucc_status_t       status    = UCC_OK;

    TASK_ARGS(task).src.info.buffer   = NULL;
    TASK_ARGS(task).src.info.count    = 0;
    TASK_ARGS(task).src.info.mem_type = UCC_MEMORY_TYPE_UNKNOWN;
    TASK_ARGS(task).src.info.datatype = UCC_DT_INT8;
    task->bcast_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.fanout_kn_radix, team_size);

    task->super.post      = ucc_tl_ucp_bcast_knomial_start;
    task->super.progress  = ucc_tl_ucp_bcast_knomial_progress;
    return status;
}
