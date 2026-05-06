/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "coll_patterns/ring.h"

#define MAX_RINGS 8

void ucc_tl_ucp_allgather_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    void               *rbuf     = args->dst.info.buffer;
    ucc_memory_type_t   rmem     = args->dst.info.mem_type;
    size_t              count    = args->dst.info.count;
    ucc_datatype_t      dt       = args->dst.info.datatype;
    size_t              rdt_size = ucc_dt_size(dt);
    ucc_ring_pattern_t *ring     = team->cuda_ring;
    ucc_rank_t          ring_id;
    ucc_rank_t          nrings;
    ucc_rank_t          rrank;
    ucc_rank_t          tsize;
    ucc_rank_t          send_idx, recv_idx, sendto, recvfrom, step;
    size_t              block_count, ring_offset, ring_count;
    size_t              data_size, data_displ;

    nrings      = ucc_min(MAX_RINGS, ring->num_rings);
    tsize       = ucc_ring_pattern_size(ring, 0);
    block_count = count / tsize;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    while (task->tagged.send_posted < 1 + nrings * (tsize - 1)) {
        ucc_assert(task->tagged.send_posted > 0);
        ucc_assert(task->tagged.recv_posted > 0);
        ucc_assert(task->tagged.send_posted == task->tagged.recv_posted);
        step = (ucc_rank_t)((task->tagged.send_posted - 1) / nrings);
        for (ring_id = 0; ring_id < nrings; ring_id++) {
            rrank    = ucc_ring_pattern_rank(ring, ring_id);
            sendto   = ucc_ring_pattern_get_send_peer(ring, ring_id, rrank);
            recvfrom = ucc_ring_pattern_get_recv_peer(ring, ring_id, rrank);

            send_idx   = ucc_ring_pattern_get_send_block(ring, ring_id,
                                                         rrank, step);
            ring_offset = ucc_buffer_block_offset(block_count, nrings,
                                                  ring_id);
            ring_count  = ucc_buffer_block_count(block_count, nrings,
                                                 ring_id);
            data_displ  = (send_idx * block_count + ring_offset) * rdt_size;
            data_size   = ring_count * rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(rbuf, data_displ),
                                             data_size, rmem, sendto, team,
                                             task),
                          task, out);

            recv_idx    = ucc_ring_pattern_get_recv_block(ring, ring_id,
                                                          rrank, step);
            ring_offset = ucc_buffer_block_offset(block_count, nrings,
                                                  ring_id);
            ring_count  = ucc_buffer_block_count(block_count, nrings,
                                                 ring_id);
            data_displ  = (recv_idx * block_count + ring_offset) * rdt_size;
            data_size   = ring_count * rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(PTR_OFFSET(rbuf, data_displ),
                                             data_size, rmem, recvfrom,
                                             team, task),
                          task, out);
        }
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_ring_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t  *team  = TASK_TEAM(task);
    ucc_coll_args_t    *args  = &TASK_ARGS(task);
    ucc_ring_pattern_t *ring  = team->cuda_ring;
    size_t              count = args->dst.info.count;
    void               *sbuf  = args->src.info.buffer;
    void               *rbuf  = args->dst.info.buffer;
    ucc_memory_type_t   rmem  = args->dst.info.mem_type;
    ucc_memory_type_t   smem  = args->src.info.mem_type;
    ucc_datatype_t      dt    = args->dst.info.datatype;
    ucc_rank_t          tsize  = ucc_ring_pattern_size(ring, 0);
    ucc_rank_t          block = UCC_TL_TEAM_RANK(team);
    size_t              data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_ring_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (!UCC_IS_INPLACE(*args)) {
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_size * block),
                              sbuf, data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    task->tagged.send_posted = task->tagged.send_completed = 1;
    task->tagged.recv_posted = task->tagged.recv_completed = 1;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allgather_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!team->cuda_ring) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_ucp_allgather_ring_start;
    task->super.progress = ucc_tl_ucp_allgather_ring_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allgather_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
