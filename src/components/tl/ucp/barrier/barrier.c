/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "barrier.h"

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_task_t *task);
void ucc_tl_ucp_barrier_knomial_progress(ucc_coll_task_t *task);

ucc_base_coll_alg_info_t
    ucc_tl_ucp_barrier_algs[UCC_TL_UCP_BARRIER_ALG_LAST + 1] = {
        [UCC_TL_UCP_BARRIER_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_BARRIER_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive knomial with arbitrary radix"},
        [UCC_TL_UCP_BARRIER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_barrier_init(ucc_tl_ucp_task_t *task)
{
    task->super.post     = ucc_tl_ucp_barrier_knomial_start;
    task->super.progress = ucc_tl_ucp_barrier_knomial_progress;
    return UCC_OK;
}
