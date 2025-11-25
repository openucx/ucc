/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "reduce.h"
#include "components/mc/ucc_mc.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_algs[UCC_TL_UCP_REDUCE_ALG_LAST + 1] = {
        [UCC_TL_UCP_REDUCE_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_REDUCE_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "reduce over knomial tree with arbitrary radix "
                     "(optimized for latency)"},
        [UCC_TL_UCP_REDUCE_ALG_DBT] =
            {.id   = UCC_TL_UCP_REDUCE_ALG_DBT,
             .name = "dbt",
             .desc = "reduce over double binary tree where a leaf in one tree "
                     "will be intermediate in other (optimized for BW)"},
        [UCC_TL_UCP_REDUCE_ALG_SRG] =
            {.id   = UCC_TL_UCP_REDUCE_ALG_SRG,
             .name = "srg",
             .desc = "recursive knomial scatter-reduce followed by knomial "
                     "gather"},
        [UCC_TL_UCP_REDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t   *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         myrank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         root      = args->root;
    ucc_rank_t         vrank     = (myrank - root + team_size) % team_size;
    ucc_status_t       status    = UCC_OK;
    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    int                isleaf;
    int                self_avg;

    if (root == myrank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
        mtype = args->dst.info.mem_type;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
        mtype = args->src.info.mem_type;
    }
    data_size = count * ucc_dt_size(dt);
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post      = ucc_tl_ucp_reduce_knomial_start;
    task->super.progress  = ucc_tl_ucp_reduce_knomial_progress;
    task->super.finalize  = ucc_tl_ucp_reduce_knomial_finalize;
    task->reduce_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_kn_radix, team_size);
    CALC_KN_TREE_DIST(team_size, task->reduce_kn.radix,
                      task->reduce_kn.max_dist);
    isleaf   = (vrank % task->reduce_kn.radix != 0 || vrank == team_size - 1);
    self_avg = (vrank % task->reduce_kn.radix == 0 && args->op == UCC_OP_AVG &&
                UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op);
    task->reduce_kn.scratch_mc_header = NULL;

    if (!isleaf || self_avg) {
    	/* scratch of size radix to fit up to radix - 1 received vectors
    	from its children at each step,
    	and an additional 1 for previous step reduce multi result */
        status = ucc_mc_alloc(&task->reduce_kn.scratch_mc_header,
                              task->reduce_kn.radix * data_size, mtype);
        task->reduce_kn.scratch =
                        task->reduce_kn.scratch_mc_header->addr;
    }

    return status;
}

ucc_status_t ucc_tl_ucp_reduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t      *team,
                                            ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task    = ucc_tl_ucp_init_task(coll_args, team);
    status  = ucc_tl_ucp_reduce_init(task);
    *task_h = &task->super;
    return status;
}
