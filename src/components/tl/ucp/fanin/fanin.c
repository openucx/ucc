/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "fanin.h"
#include "../reduce/reduce.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_fanin_algs[UCC_TL_UCP_FANIN_ALG_LAST + 1] = {
        [UCC_TL_UCP_FANIN_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_FANIN_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "fanin over knomial tree with arbitrary radix"},
        [UCC_TL_UCP_FANIN_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_fanin_init(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         team_size = UCC_TL_TEAM_SIZE(team);
    ucc_status_t       status    = UCC_OK;

    TASK_ARGS(task).src.info.buffer   = NULL;
    TASK_ARGS(task).src.info.count    = 0;
    TASK_ARGS(task).src.info.mem_type = UCC_MEMORY_TYPE_UNKNOWN;
    TASK_ARGS(task).src.info.datatype = UCC_DT_INT8;

    TASK_ARGS(task).dst.info.buffer   = NULL;
    TASK_ARGS(task).dst.info.count    = 0;
    TASK_ARGS(task).dst.info.mem_type = UCC_MEMORY_TYPE_UNKNOWN;
    TASK_ARGS(task).dst.info.datatype = UCC_DT_INT8;

    task->super.post      = ucc_tl_ucp_reduce_knomial_start;
    task->super.progress  = ucc_tl_ucp_reduce_knomial_progress;
    task->reduce_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.fanin_kn_radix, team_size);

    CALC_KN_TREE_DIST(team_size, task->reduce_kn.radix,
                      task->reduce_kn.max_dist);
    return status;
}
