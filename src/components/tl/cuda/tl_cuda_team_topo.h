/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_TEAM_TOPO_H_
#define UCC_TL_CUDA_TEAM_TOPO_H_

#include "components/tl/ucc_tl.h"
#include "tl_cuda_topo.h"

typedef struct ucc_tl_cuda_proxy {
    ucc_rank_t src;   /* source rank */
    ucc_rank_t dst;   /* destination rank */
    ucc_rank_t proxy; /* proxy rank */
} ucc_tl_cuda_proxy_t;

typedef struct ucc_tl_cuda_ring {
    ucc_rank_t *ring;  /* list of ranks forming ring */
    ucc_rank_t *iring; /* inverse of ring */
} ucc_tl_cuda_ring_t;

typedef struct ucc_tl_cuda_team_topo {
    ucc_rank_t          *matrix;             /* nvlink adjacency matrix */
    int                  proxy_needed;       /* is proxy needed for current rank */
    int                  num_proxies;        /* number of entries in proxies list */
    ucc_tl_cuda_proxy_t *proxies;            /* list of pairs where current rank is proxy */
    int                  num_rings;          /* number of entries in rings list */
    ucc_tl_cuda_ring_t  *rings;              /* list of rings for ring algorithms */
    int                  is_fully_connected; /* no proxies in team topo */
} ucc_tl_cuda_team_topo_t;

ucc_status_t ucc_tl_cuda_team_topo_create(const ucc_tl_team_t *team,
                                          ucc_tl_cuda_team_topo_t **team_topo);

ucc_status_t ucc_tl_cuda_team_topo_destroy(ucc_tl_cuda_team_topo_t *team_topo);

void ucc_tl_cuda_team_topo_print_proxies(const ucc_tl_team_t *team,
                                         const ucc_tl_cuda_team_topo_t *topo);

void ucc_tl_cuda_team_topo_print_rings(const ucc_tl_team_t *tl_team,
                                       const ucc_tl_cuda_team_topo_t *topo);

static inline int
ucc_tl_cuda_team_topo_is_direct(const ucc_tl_team_t *team,
                                const ucc_tl_cuda_team_topo_t *topo,
                                ucc_rank_t r1, ucc_rank_t r2)
{
    return topo->matrix[r1 * team->super.params.size + r2] != 0;
}

static inline int
ucc_tl_cuda_team_topo_is_fully_connected(const ucc_tl_cuda_team_topo_t *topo)
{
    return topo->is_fully_connected;
}

#endif
