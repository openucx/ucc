/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task)
{
    if ((!UCC_DT_IS_PREDEFINED((TASK_ARGS(task)).dst.info.datatype)) ||
        (!UCC_IS_INPLACE(TASK_ARGS(task)) &&
         (!UCC_DT_IS_PREDEFINED((TASK_ARGS(task)).src.info.datatype)))) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_ucp_allgather_ring_start;
    task->super.progress = ucc_tl_ucp_allgather_ring_progress;
    return UCC_OK;
}
