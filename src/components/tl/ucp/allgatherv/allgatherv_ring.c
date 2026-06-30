/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/ring.h"


#define MAX_RINGS 8

void ucc_tl_ucp_allgatherv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ucc_ring_pattern_t *ring     = team->cuda_ring;
    ucc_rank_t          ring_id;
    ucc_rank_t          nrings;
    ucc_rank_t          rrank;
    ptrdiff_t           rbuf     = (ptrdiff_t)args->dst.info_v.buffer;
    ucc_memory_type_t   rmem     = args->dst.info_v.mem_type;
    size_t              rdt_size = ucc_dt_size(args->dst.info_v.datatype);
    ucc_rank_t          send_idx, recv_idx, sendto, recvfrom, step, tsize;
    size_t              data_size, data_displ;
    size_t              block_count, ring_count, ring_offset, base_displ;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    nrings = ucc_min(MAX_RINGS, ring->num_rings);
    tsize  = ucc_ring_pattern_size(ring, 0);

    while (task->tagged.send_posted < 1 + nrings * (tsize - 1)) {
        ucc_assert(task->tagged.send_posted > 0);
        ucc_assert(task->tagged.recv_posted > 0);
        ucc_assert(task->tagged.send_posted == task->tagged.recv_posted);
        step = (ucc_rank_t)((task->tagged.send_posted - 1) / nrings);
        for (ring_id = 0; ring_id < nrings; ring_id++) {

            rrank    = ucc_ring_pattern_rank(ring, ring_id);
            sendto   = ucc_ring_pattern_get_send_peer(ring, ring_id, rrank);
            recvfrom = ucc_ring_pattern_get_recv_peer(ring, ring_id, rrank);

            send_idx = ucc_ring_pattern_get_send_block(ring, ring_id, rrank,
                                                       step);
            block_count = ucc_coll_args_get_count(
                args, args->dst.info_v.counts, send_idx);
            ring_offset = ucc_buffer_block_offset(block_count, nrings, ring_id);
            ring_count  = ucc_buffer_block_count(block_count, nrings, ring_id);
            base_displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, send_idx);
            data_displ = (base_displ + ring_offset) * rdt_size;
            data_size  = ring_count * rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(rbuf + data_displ),
                                             data_size, rmem, sendto, team,
                                             task),
                          task, out);

            recv_idx = ucc_ring_pattern_get_recv_block(ring, ring_id, rrank,
                                                       step);
            block_count = ucc_coll_args_get_count(
                args, args->dst.info_v.counts, recv_idx);
            ring_offset = ucc_buffer_block_offset(block_count, nrings, ring_id);
            ring_count  = ucc_buffer_block_count(block_count, nrings, ring_id);
            base_displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, recv_idx);
            data_displ = (base_displ + ring_offset) * rdt_size;
            data_size  = ring_count * rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + data_displ),
                                             data_size, rmem, recvfrom, team,
                                             task),
                          task, out);
        }
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    return;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    void *             sbuf  = args->src.info.buffer;
    void *             rbuf  = args->dst.info_v.buffer;
    ucc_memory_type_t  smem  = args->src.info.mem_type;
    ucc_memory_type_t  rmem  = args->dst.info_v.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    size_t             data_size, data_displ, rdt_size;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (!UCC_IS_INPLACE(*args)) {
        /* TODO replace local sendrecv with memcpy? */
        rdt_size   = ucc_dt_size(args->dst.info_v.datatype);
        data_displ = ucc_coll_args_get_displacement(args,
                         args->dst.info_v.displacements, grank) * rdt_size;
        data_size =  ucc_coll_args_get_count(args,
                         args->dst.info_v.counts, grank) * rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(PTR_OFFSET(rbuf, data_displ), data_size,
                                         rmem, grank, team, task),
                      task, error);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(sbuf, data_size, smem, grank, team, task),
                      task, error);
    } else {
        /* to simplify progress fucnction and make it identical for
           in-place and non in-place */
        task->tagged.send_posted = task->tagged.send_completed = 1;
        task->tagged.recv_posted = task->tagged.recv_completed = 1;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
error:
    return task->super.status;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }

    task->super.post     = ucc_tl_ucp_allgatherv_ring_start;
    task->super.progress = ucc_tl_ucp_allgatherv_ring_progress;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team,
                                             ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allgatherv_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
