/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "reduce.h"

ucc_status_t ucc_tl_ucp_reduce_knomial_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_reduce_knomial_progress(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_reduce_knomial_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task)
{
    task->super.post     = ucc_tl_ucp_reduce_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_knomial_finalize;
    ucc_coll_args_t   *args      = &task->super.args;
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         myrank    = team->rank;
    ucc_rank_t         team_size = team->size;
    ucc_rank_t         root      = (uint32_t)args->root;
    ucc_rank_t         vrank     = (myrank - root + team_size) % team_size;
    ucc_memory_type_t  mtype     = args->src.info.mem_type;
    ucc_status_t       status    = UCC_OK;
    size_t             data_size =
        args->src.info.count * ucc_dt_size(args->src.info.datatype);
    int                isleaf;

    task->reduce_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_kn_radix, team->size);
    isleaf = (vrank % task->reduce_kn.radix != 0 || vrank == team_size - 1);

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
