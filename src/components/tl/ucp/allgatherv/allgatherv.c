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
ucc_status_t ucc_tl_ucp_allgatherv_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task)
{
    if ((task->args.dst.info_v.datatype == UCC_DT_USERDEFINED) ||
        (!UCC_IS_INPLACE(task->args) &&
         (task->args.src.info.datatype == UCC_DT_USERDEFINED))) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_ucp_allgatherv_ring_start;
    task->super.progress = ucc_tl_ucp_allgatherv_ring_progress;
    return UCC_OK;
}
