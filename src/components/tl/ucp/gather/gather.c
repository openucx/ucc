/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static inline uint32_t calc_buffer_size(ucc_rank_t rank, uint32_t radix, ucc_rank_t team_size)
{
    uint32_t radix_valuation;

    if (rank == 0) {
        return team_size;
    }
    radix_valuation = calc_valuation(rank, radix);
    return (uint32_t)ucc_min(pow(radix, radix_valuation), team_size - rank);
}

ucc_status_t ucc_tl_ucp_gather_init(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t *  args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         myrank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         root      = args->root;
    ucc_rank_t         vrank     = (myrank - root + team_size) % team_size;
    ucc_status_t       status    = UCC_OK;
    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    uint32_t           buffer_size;
    int                isleaf;

    if (root == myrank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
        mtype = args->dst.info.mem_type;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
        mtype = args->src.info.mem_type;
    }
    data_size            = count * ucc_dt_size(dt);
    task->super.post     = ucc_tl_ucp_gather_knomial_start;
    task->super.progress = ucc_tl_ucp_gather_knomial_progress;
    task->super.finalize = ucc_tl_ucp_gather_knomial_finalize;
    task->gather_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.gather_kn_radix, team_size);
    CALC_KN_TREE_DIST(team_size, task->gather_kn.radix,
                      task->gather_kn.max_dist);
    isleaf = (vrank % task->gather_kn.radix != 0 || vrank == team_size - 1);
    task->gather_kn.scratch_mc_header = NULL;

    if (vrank == 0) {
        task->gather_kn.scratch = args->dst.info.buffer;
    } else if (isleaf) {
        task->gather_kn.scratch = args->src.info.buffer;
    } else {
        buffer_size = calc_buffer_size(vrank, task->gather_kn.radix, team_size);
        status      = ucc_mc_alloc(&task->gather_kn.scratch_mc_header,
                              buffer_size * data_size, mtype);
        task->gather_kn.scratch = task->gather_kn.scratch_mc_header->addr;
    }

    return status;
}
