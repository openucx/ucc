/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_shm_coll_perf_params.h"

void ucc_tl_shm_perf_params_amd_rome_128_bcast(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    size_t             data_size = ucc_coll_args_msgsize(&task->super.bargs);

    if (TASK_LIB(task)->cfg.set_perf_params) {
        if (data_size < 256) {
            task->progress_alg   = BCAST_WW;
            task->base_tree_only = 1;
            task->base_radix     = 4;
            task->top_radix      = 0;
        } else {
            task->progress_alg   = BCAST_WR;
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
    size_t             data_size = ucc_coll_args_msgsize(&task->super.bargs);

    if (TASK_LIB(task)->cfg.set_perf_params) {
        if (data_size < 256) {
            task->base_tree_only = 0;
            task->base_radix     = 4;
            task->top_radix      = TASK_LIB(task)->cfg.reduce_top_radix;
        } else {
            task->base_tree_only = 0;
            task->base_radix     = 2;
            task->top_radix      = TASK_LIB(task)->cfg.reduce_top_radix;
        }
    } else {
        ucc_tl_shm_perf_params_generic_reduce(coll_task);
    }
}
