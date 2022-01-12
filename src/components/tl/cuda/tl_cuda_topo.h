/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_TOPO_H_
#define UCC_TL_CUDA_TOPO_H_

#include "tl_cuda.h"

ucc_status_t ucc_tl_cuda_topo_create(ucc_tl_cuda_team_t *team,
                                     ucc_tl_cuda_topo_t **cuda_topo);

ucc_status_t ucc_tl_cuda_topo_destroy(ucc_tl_cuda_topo_t *cuda_topo);

static inline int ucc_tl_cuda_topo_is_direct(ucc_tl_cuda_team_t *team,
                                             ucc_tl_cuda_topo_t *cuda_topo,
                                             ucc_rank_t r1, ucc_rank_t r2)
{
    return cuda_topo->matrix[r1 * UCC_TL_TEAM_SIZE(team) + r2] == 1;
}

void ucc_tl_cuda_topo_print(ucc_tl_cuda_team_t *team,
                            ucc_tl_cuda_topo_t *cuda_topo);

#endif
