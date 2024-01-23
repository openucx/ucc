/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allreduce_algs[UCC_TL_UCP_ALLREDUCE_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive knomial with arbitrary radix (optimized for "
                     "latency)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL,
             .name = "sra_knomial",
             .desc = "recursive knomial scatter-reduce followed by knomial "
                     "allgather (optimized for BW)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW,
             .name = "sliding_window",
             .desc = "sliding window allreduce (optimized for running on DPU)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;
    ALLREDUCE_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
out:
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *     team,
                                         ucc_coll_task_t **    task_h)
{
    ucc_status_t             status  = UCC_OK;
    ucc_tl_ucp_team_t *      tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *      task;
    ucc_ee_executor_params_t params;

    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        ucc_error("couldnt allocate task");
        return UCC_ERR_NO_MEMORY;
    }
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_allreduce_sliding_window_start;
    task->super.progress = ucc_tl_ucp_allreduce_sliding_window_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_sliding_window_finalize;

    ucc_tl_ucp_allreduce_sliding_window_task_init(coll_args, team, task);

    params.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
    params.ee_type = UCC_EE_CPU_THREAD;
    status =
        ucc_ee_executor_init(&params, &task->allreduce_sliding_window.executor);

    if (UCC_OK != status) {
        ucc_error("failed to init executor: %s", ucc_status_string(status));
    }

out:
    return status;
}
