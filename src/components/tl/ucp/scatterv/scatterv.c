/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "scatterv.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_tl_ucp_scatterv_linear_start(ucc_coll_task_t *task);

void ucc_tl_ucp_scatterv_linear_progress(ucc_coll_task_t *task);

ucc_base_coll_alg_info_t
    ucc_tl_ucp_scatterv_algs[UCC_TL_UCP_SCATTERV_ALG_LAST + 1] = {
        [UCC_TL_UCP_SCATTERV_ALG_LINEAR] =
            {.id   = UCC_TL_UCP_SCATTERV_ALG_LINEAR,
             .name = "linear",
             .desc = "linear scatterv algorithm"},
        [UCC_TL_UCP_SCATTERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_scatterv_linear_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_scatterv_init(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_rank_t         trank = UCC_TL_TEAM_RANK(team);

    if (UCC_IS_ROOT(*args, trank)) {
        if (!UCC_DT_IS_PREDEFINED(args->src.info_v.datatype)) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    if (!UCC_IS_ROOT(*args, trank) || !UCC_IS_INPLACE(*args)) {
        if (!UCC_DT_IS_PREDEFINED(args->dst.info.datatype)) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    return ucc_tl_ucp_scatterv_linear_init(task);
}
