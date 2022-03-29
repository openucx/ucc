/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "barrier.h"

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_task_t *task);
void ucc_tl_ucp_barrier_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_barrier_init(ucc_tl_ucp_task_t *task)
{
    task->super.post     = ucc_tl_ucp_barrier_knomial_start;
    task->super.progress = ucc_tl_ucp_barrier_knomial_progress;
    return UCC_OK;
}
