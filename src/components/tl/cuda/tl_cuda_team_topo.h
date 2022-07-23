/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_TEAM_TOPO_H_
#define UCC_TL_CUDA_TEAM_TOPO_H_

#include "components/tl/ucc_tl.h"
#include "tl_cuda_topo.h"

typedef struct ucc_tl_cuda_proxy {
    ucc_rank_t src;
    ucc_rank_t dst;
    ucc_rank_t proxy;
} ucc_tl_cuda_proxy_t;

typedef struct ucc_tl_cuda_ring {
    ucc_rank_t *ring;
    ucc_rank_t *iring;
} ucc_tl_cuda_ring_t;

typedef struct ucc_tl_cuda_team_topo {
    int                     *matrix;
    int                      num_proxies;
    ucc_tl_cuda_proxy_t     *proxies;
    int                      num_rings;
    ucc_tl_cuda_ring_t      *rings;
} ucc_tl_cuda_team_topo_t;

ucc_status_t ucc_tl_cuda_team_topo_create(const ucc_tl_team_t *team,
                                          ucc_tl_cuda_team_topo_t **team_topo);

ucc_status_t ucc_tl_cuda_team_topo_destroy(ucc_tl_cuda_team_topo_t *team_topo);

void ucc_tl_cuda_team_topo_print(const ucc_tl_team_t *team,
                                 const ucc_tl_cuda_team_topo_t *cuda_topo);

void ucc_tl_cuda_team_topo_print_rings(const ucc_tl_team_t *tl_team,
                                       const ucc_tl_cuda_team_topo_t *topo);

static inline int
ucc_tl_cuda_team_topo_is_direct(const ucc_tl_team_t *team,
                                const ucc_tl_cuda_team_topo_t *topo,
                                ucc_rank_t r1, ucc_rank_t r2)
{
    return topo->matrix[r1 * team->super.params.size + r2] != 0;
}

#endif
