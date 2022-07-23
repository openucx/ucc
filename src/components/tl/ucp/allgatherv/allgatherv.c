/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_tl_ucp_allgatherv_ring_start(ucc_coll_task_t *task);

void ucc_tl_ucp_allgatherv_ring_progress(ucc_coll_task_t *task);

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgatherv_algs[UCC_TL_UCP_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLGATHERV_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLGATHERV_ALG_RING,
             .name = "ring",
             .desc = "O(N) Ring"},
        [UCC_TL_UCP_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

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
