/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "gatherv.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_tl_ucp_gatherv_linear_start(ucc_coll_task_t *task);

void ucc_tl_ucp_gatherv_linear_progress(ucc_coll_task_t *task);

ucc_base_coll_alg_info_t
    ucc_tl_ucp_gatherv_algs[UCC_TL_UCP_GATHERV_ALG_LAST + 1] = {
        [UCC_TL_UCP_GATHERV_ALG_LINEAR] =
            {.id   = UCC_TL_UCP_GATHERV_ALG_LINEAR,
             .name = "linear",
             .desc = "linear gatherv algorithm"},
        [UCC_TL_UCP_GATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_gatherv_linear_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_gatherv_init(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);

    if (UCC_TL_TEAM_RANK(team) == args->root) {
        if (!UCC_DT_IS_PREDEFINED(args->dst.info_v.datatype)) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }
    if ((UCC_TL_TEAM_RANK(team) != args->root) ||
        (!UCC_IS_INPLACE(*args))) {
        if (!UCC_DT_IS_PREDEFINED(args->src.info.datatype)) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    return ucc_tl_ucp_gatherv_linear_init(task);
}
