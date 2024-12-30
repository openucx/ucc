/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"
#include "components/mc/ucc_mc.h"

ucc_base_coll_alg_info_t
    ucc_tl_cuda_bcast_algs[UCC_TL_CUDA_BCAST_ALG_LAST + 1] = {
        [UCC_TL_CUDA_BCAST_ALG_LINEAR] = {.id   = UCC_TL_CUDA_BCAST_ALG_LINEAR,
                                          .name = "linear",
                                          .desc = "linear bcast algorithm"},
        [UCC_TL_CUDA_BCAST_ALG_LAST]   = {.id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_cuda_bcast_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t      *tl_team,
                                    ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);

    if (ucc_tl_cuda_team_topo_is_fully_connected(team->topo)) {
        return ucc_tl_cuda_bcast_linear_init(coll_args, tl_team, task_p);
    } else {
        return UCC_ERR_NOT_SUPPORTED;
    }
}
