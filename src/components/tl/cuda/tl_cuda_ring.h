/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static inline int get_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                int ring_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);

    return sync->seq_num[ring_id];
}

static inline void set_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                 int step, int ring_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);

    sync->seq_num[ring_id] = step;
}

#endif
