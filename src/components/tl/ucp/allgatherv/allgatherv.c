/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_tl_ucp_allgatherv_ring_start(ucc_coll_task_t *task);

void ucc_tl_ucp_allgatherv_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task)
{
    if ((!UCC_DT_IS_PREDEFINED((TASK_ARGS(task)).dst.info_v.datatype)) ||
        (!UCC_IS_INPLACE(TASK_ARGS(task)) &&
         (!UCC_DT_IS_PREDEFINED((TASK_ARGS(task)).src.info.datatype)))) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_ucp_allgatherv_ring_start;
    task->super.progress = ucc_tl_ucp_allgatherv_ring_progress;
    return UCC_OK;
}
