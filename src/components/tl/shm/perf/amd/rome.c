/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

void ucc_tl_shm_perf_params_amd_rome_128_bcast(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size;

    data_size = args.src.info.count * ucc_dt_size(args.src.info.datatype);

    if (TASK_LIB(task)->cfg.set_perf_params) {
        if (data_size < 256) {
            task->progress_alg   = 0; // WRITE
            task->base_tree_only = 1;
            task->base_radix     = 4;
            task->top_radix      = 0;
        } else {
            task->progress_alg   = 1; // wr
            task->base_tree_only = 0;
            task->base_radix     = 16;
            task->top_radix      = TASK_LIB(task)->cfg.bcast_top_radix;
        }
    } else {
        ucc_tl_shm_perf_params_generic_bcast(coll_task);
    }
}

void ucc_tl_shm_perf_params_amd_rome_128_reduce(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    size_t             data_size;

    if (rank == args.root) {
        data_size = args.dst.info.count * ucc_dt_size(args.dst.info.datatype);
    } else {
        data_size = args.src.info.count * ucc_dt_size(args.src.info.datatype);
    }

    if (TASK_LIB(task)->cfg.set_perf_params) {
        if (data_size < 256) {
            task->base_tree_only = 0; // READ
            task->base_radix     = 4;
            task->top_radix      = TASK_LIB(task)->cfg.reduce_top_radix;
        } else {
            task->base_tree_only = 0; // READ
            task->base_radix     = 2;
            task->top_radix      = TASK_LIB(task)->cfg.reduce_top_radix;
        }
    } else {
        ucc_tl_shm_perf_params_generic_reduce(coll_task);
    }
}
