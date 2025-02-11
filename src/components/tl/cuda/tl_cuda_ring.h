/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_RING_H_
#define UCC_TL_CUDA_RING_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum {
    RING_STAGE_SYNC,    /*< Wait for free SYNC segment */
    RING_STAGE_SETUP,   /*< Wait for memhandle setup to finish */
    RING_STAGE_RING,    /*< Ring algorithm is running */
    RING_STAGE_BARRIER, /*< Ring algorithm is done, waiting for
                         *  other ranks to finish
                         */
};

#define RING_MIN_MSG_SIZE 8192  /* Min size of msg per ring */

static inline size_t get_scratch_size(ucc_tl_cuda_team_t *team, int nrings,
                                      int nchunks, size_t dt_size)
{
    size_t ratio = 2 * nrings * nchunks * dt_size;

    return ucc_align_down_pow2((UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size /
                                ratio), 64) * ratio;
}

static inline int get_num_rings(ucc_tl_cuda_team_t *team, size_t msgsize,
                                unsigned long rings_requested)
{
    int nrings;

    if (rings_requested != UCC_ULUNITS_AUTO) {
        nrings = ucc_min(rings_requested, team->topo->num_rings);
    } else {
        nrings = team->topo->num_rings;
    }

    nrings = ucc_min(nrings, UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS);
    nrings = ucc_min(nrings, ucc_div_round_up(msgsize, RING_MIN_MSG_SIZE));

    return nrings;
}

static inline ucc_rank_t get_send_to(ucc_tl_cuda_team_t *team,
                                     ucc_rank_t trank, ucc_rank_t tsize,
                                     int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] + 1) % tsize];
}

static inline ucc_rank_t get_recv_from(ucc_tl_cuda_team_t *team,
                                       ucc_rank_t trank, ucc_rank_t tsize,
                                       int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] - 1 + tsize) % tsize];
}

static inline ucc_rank_t get_send_block(ucc_tl_cuda_team_t *team,
                                        ucc_rank_t trank, ucc_rank_t tsize,
                                        uint32_t step, int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] + tsize - step) % tsize];
}

static inline ucc_rank_t get_recv_block(ucc_tl_cuda_team_t *team,
                                        ucc_rank_t trank, ucc_rank_t tsize,
                                        uint32_t step, int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] + tsize - step - 1) % tsize];
}

#endif
