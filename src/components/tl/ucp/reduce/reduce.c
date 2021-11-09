/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "reduce.h"

ucc_status_t ucc_tl_ucp_reduce_knomial_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_reduce_knomial_progress(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_reduce_knomial_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t   *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         myrank    = team->rank;
    ucc_rank_t         team_size = team->size;
    ucc_rank_t         root      = args->root;
    ucc_rank_t         vrank     = (myrank - root + team_size) % team_size;
    ucc_status_t       status    = UCC_OK;
    ucc_memory_type_t  mtype;
    size_t             data_size;
    int                isleaf;

    if (root == myrank) {
        data_size = args->dst.info.count * ucc_dt_size(args->dst.info.datatype);
        mtype = args->dst.info.mem_type;
    } else {
        data_size = args->src.info.count * ucc_dt_size(args->src.info.datatype);
        mtype = args->src.info.mem_type;
    }
    task->super.post      = ucc_tl_ucp_reduce_knomial_start;
    task->super.progress  = ucc_tl_ucp_reduce_knomial_progress;
    task->super.finalize  = ucc_tl_ucp_reduce_knomial_finalize;
    task->reduce_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_kn_radix, team->size);
    CALC_KN_TREE_DIST(team->size, task->reduce_kn.radix,
                      task->reduce_kn.max_dist);
    isleaf = (vrank % task->reduce_kn.radix != 0 || vrank == team_size - 1);
    task->reduce_kn.scratch_mc_header = NULL;

    if (!isleaf) {
    	/* scratch of size radix to fit up to radix - 1 recieved vectors
    	from its children at each step,
    	and an additional 1 for previous step reduce multi result */
        status = ucc_mc_alloc(&task->reduce_kn.scratch_mc_header,
                              task->reduce_kn.radix * data_size, mtype);
        task->reduce_kn.scratch =
                        task->reduce_kn.scratch_mc_header->addr;
    }
    return status;
}
