/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_tl_ucp_allreduce_knomial_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task)
{
    if (task->args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "userdefined reductions are not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (!UCC_IS_INPLACE(task->args) && (task->args.src.info.mem_type !=
                                        task->args.dst.info.mem_type)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "assymetric src/dst memory types are not supported yetpp");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_ucp_allreduce_knomial_start;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    return UCC_OK;
}
