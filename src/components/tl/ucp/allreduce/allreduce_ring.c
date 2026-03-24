/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_dt_reduce.h"
#include "coll_patterns/ring.h"

#define MAX_RINGS 8

/* allreduce ring = reduce_scatter + allgather
 *
 * Phase 0: reduce_scatter: each rank accumulates reductions and ends up
 *         with its own reduced block in dst.
 * Phase 1: allgather:      in-place ring allgather distributes all blocks.
 *
 * Phase is tracked via s_scratch_busy[0] (0 = RS, 1 = AG).
 */

static void
ucc_tl_ucp_allreduce_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task      = ucc_derived_of(coll_task,
                                                    ucc_tl_ucp_task_t);
    ucc_coll_args_t    *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t  *team      = TASK_TEAM(task);
    ucc_ring_pattern_t *ring      = team->cuda_ring;
    ucc_rank_t          nrings    = ucc_min(MAX_RINGS, ring->num_rings);
    ucc_rank_t          tsize     = ucc_ring_pattern_size(ring, 0);
    size_t              total_cnt = args->dst.info.count;
    size_t              block_cnt = total_cnt / tsize;
    void               *dst       = args->dst.info.buffer;
    void               *scratch   = task->reduce_scatter_ring.scratch;
    ucc_memory_type_t   mem_type  = args->dst.info.mem_type;
    ucc_datatype_t      dt        = args->dst.info.datatype;
    size_t              dt_size   = ucc_dt_size(dt);
    ucc_rank_t          rrank, adj_rrank, ring_id, step;
    ucc_rank_t          sendto, recvfrom, send_block, recv_block;
    size_t              ring_offset, ring_count, data_displ, data_size;
    ucc_status_t        status;
    int                 is_avg;

    if (task->reduce_scatter_ring.s_scratch_busy[0] == 0) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }

        while (task->tagged.send_posted < 1 + nrings * (tsize - 1)) {
            ucc_assert(task->tagged.send_posted > 0);
            ucc_assert(task->tagged.recv_posted > 0);

            step   = (ucc_rank_t)((task->tagged.send_posted - 1) / nrings);
            is_avg = (args->op == UCC_OP_AVG) &&
                     (step == (ucc_rank_t)(tsize - 2));

            for (ring_id = 0; ring_id < nrings; ring_id++) {
                rrank       = ucc_ring_pattern_rank(ring, ring_id);
                adj_rrank   = (rrank + tsize - 1) % tsize;
                recv_block  = ucc_ring_pattern_get_recv_block(ring, ring_id,
                                                              adj_rrank, step);
                ring_offset = ucc_buffer_block_offset(block_cnt, nrings,
                                                      ring_id);
                ring_count  = ucc_buffer_block_count(block_cnt, nrings,
                                                     ring_id);

                status = ucc_dt_reduce(
                    PTR_OFFSET(scratch, ring_offset * dt_size),
                    PTR_OFFSET(dst,
                               (recv_block * block_cnt + ring_offset) *
                                   dt_size),
                    PTR_OFFSET(dst,
                               (recv_block * block_cnt + ring_offset) *
                                   dt_size),
                    ring_count, dt, args,
                    is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                    AVG_ALPHA(task), task->reduce_scatter_ring.executor,
                    &task->reduce_scatter_ring.etask);
                if (ucc_unlikely(UCC_OK != status)) {
                    tl_error(UCC_TASK_LIB(task),
                             "failed to perform dt reduction");
                    task->super.status = status;
                    return;
                }
                EXEC_TASK_WAIT(task->reduce_scatter_ring.etask);
            }

            if (step + 1 >= (ucc_rank_t)(tsize - 1)) {
                task->reduce_scatter_ring.s_scratch_busy[0] = 1;
                task->tagged.send_posted    = 1;
                task->tagged.send_completed = 1;
                task->tagged.recv_posted    = 1;
                task->tagged.recv_completed = 1;
                goto ag_phase;
            }

            step++;
            for (ring_id = 0; ring_id < nrings; ring_id++) {
                rrank       = ucc_ring_pattern_rank(ring, ring_id);
                adj_rrank   = (rrank + tsize - 1) % tsize;
                sendto      = ucc_ring_pattern_get_send_peer(ring, ring_id,
                                                             rrank);
                recvfrom    = ucc_ring_pattern_get_recv_peer(ring, ring_id,
                                                             rrank);
                send_block  = ucc_ring_pattern_get_send_block(ring, ring_id,
                                                              adj_rrank, step);
                ring_offset = ucc_buffer_block_offset(block_cnt, nrings,
                                                      ring_id);
                ring_count  = ucc_buffer_block_count(block_cnt, nrings,
                                                     ring_id);

                data_displ = (send_block * block_cnt + ring_offset) * dt_size;
                data_size  = ring_count * dt_size;
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(
                    PTR_OFFSET(dst, data_displ), data_size, mem_type,
                    sendto, team, task), task, out);

                data_displ = ring_offset * dt_size;
                data_size  = ring_count * dt_size;
                UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(
                    PTR_OFFSET(scratch, data_displ), data_size, mem_type,
                    recvfrom, team, task), task, out);
            }

            if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
                return;
            }
        }
    }

ag_phase:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    while (task->tagged.send_posted < 1 + nrings * (tsize - 1)) {
        ucc_assert(task->tagged.send_posted > 0);
        ucc_assert(task->tagged.recv_posted > 0);
        ucc_assert(task->tagged.send_posted == task->tagged.recv_posted);

        step = (ucc_rank_t)((task->tagged.send_posted - 1) / nrings);

        for (ring_id = 0; ring_id < nrings; ring_id++) {
            ucc_rank_t send_idx, recv_idx;

            rrank    = ucc_ring_pattern_rank(ring, ring_id);
            sendto   = ucc_ring_pattern_get_send_peer(ring, ring_id, rrank);
            recvfrom = ucc_ring_pattern_get_recv_peer(ring, ring_id, rrank);

            send_idx    = ucc_ring_pattern_get_send_block(ring, ring_id,
                                                          rrank, step);
            ring_offset = ucc_buffer_block_offset(block_cnt, nrings, ring_id);
            ring_count  = ucc_buffer_block_count(block_cnt, nrings, ring_id);
            data_displ  = (send_idx * block_cnt + ring_offset) * dt_size;
            data_size   = ring_count * dt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(dst, data_displ),
                                             data_size, mem_type, sendto,
                                             team, task),
                          task, out);

            recv_idx    = ucc_ring_pattern_get_recv_block(ring, ring_id,
                                                          rrank, step);
            ring_offset = ucc_buffer_block_offset(block_cnt, nrings, ring_id);
            ring_count  = ucc_buffer_block_count(block_cnt, nrings, ring_id);
            data_displ  = (recv_idx * block_cnt + ring_offset) * dt_size;
            data_size   = ring_count * dt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(PTR_OFFSET(dst, data_displ),
                                             data_size, mem_type, recvfrom,
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
    return;
}

static ucc_status_t
ucc_tl_ucp_allreduce_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task     = ucc_derived_of(coll_task,
                                                   ucc_tl_ucp_task_t);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ucc_ring_pattern_t *ring     = team->cuda_ring;
    ucc_rank_t          nrings   = ucc_min(MAX_RINGS, ring->num_rings);
    ucc_rank_t          tsize    = ucc_ring_pattern_size(ring, 0);
    size_t              total_cnt = args->dst.info.count;
    size_t              block_cnt = total_cnt / tsize;
    ucc_datatype_t      dt       = args->dst.info.datatype;
    size_t              dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t   mem_type = args->dst.info.mem_type;
    void               *dst      = args->dst.info.buffer;
    void               *scratch  = task->reduce_scatter_ring.scratch;
    ucc_rank_t          ring_id, rrank, adj_rrank, send_block, sendto,
                        recvfrom;
    size_t              ring_offset, ring_count, data_displ, data_size;
    ucc_status_t        status;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (!UCC_IS_INPLACE(*args)) {
        status = ucc_mc_memcpy(dst, args->src.info.buffer,
                               total_cnt * dt_size, mem_type,
                               args->src.info.mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    status = ucc_coll_task_get_executor(&task->super,
                                        &task->reduce_scatter_ring.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->reduce_scatter_ring.s_scratch_busy[0] = 0;

    for (ring_id = 0; ring_id < nrings; ring_id++) {
        rrank       = ucc_ring_pattern_rank(ring, ring_id);
        adj_rrank   = (rrank + tsize - 1) % tsize;
        send_block  = ucc_ring_pattern_get_send_block(ring, ring_id,
                                                      adj_rrank, 0);
        sendto      = ucc_ring_pattern_get_send_peer(ring, ring_id, rrank);
        recvfrom    = ucc_ring_pattern_get_recv_peer(ring, ring_id, rrank);
        ring_offset = ucc_buffer_block_offset(block_cnt, nrings, ring_id);
        ring_count  = ucc_buffer_block_count(block_cnt, nrings, ring_id);

        data_displ = (send_block * block_cnt + ring_offset) * dt_size;
        data_size  = ring_count * dt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(
            PTR_OFFSET(dst, data_displ), data_size, mem_type, sendto,
            team, task), task, err);

        data_displ = ring_offset * dt_size;
        data_size  = ring_count * dt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(
            PTR_OFFSET(scratch, data_displ), data_size, mem_type, recvfrom,
            team, task), task, err);
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq,
                                      &task->super);
err:
    return task->super.status;
}

static ucc_status_t
ucc_tl_ucp_allreduce_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    ucc_mc_free(task->reduce_scatter_ring.scratch_mc_header);
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t
ucc_tl_ucp_allreduce_ring_init(ucc_base_coll_args_t *coll_args,
                               ucc_base_team_t      *team,
                               ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t     *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_coll_args_t        *args   = &coll_args->args;
    ucc_tl_ucp_task_t      *task;
    ucc_rank_t              tsize;
    size_t                  total_count, block_count, scratch_size;
    ucc_datatype_t          dt;
    size_t                  dt_size;
    ucc_memory_type_t       mem_type;
    ucc_mc_buffer_header_t *scratch_mc_header;
    ucc_status_t            status;

    if (!tl_team->cuda_ring) {
        return UCC_ERR_NOT_FOUND;
    }

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!UCC_IS_INPLACE(*args) &&
        args->src.info.mem_type != args->dst.info.mem_type) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_avg_pre_op &&
        args->op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    tsize       = ucc_ring_pattern_size(tl_team->cuda_ring, 0);
    total_count = args->dst.info.count;
    if (total_count % tsize != 0) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    block_count  = total_count / tsize;
    dt           = args->dst.info.datatype;
    dt_size      = ucc_dt_size(dt);
    mem_type     = args->dst.info.mem_type;
    scratch_size = block_count * dt_size;

    task = ucc_tl_ucp_init_task(coll_args, team);

    status = ucc_mc_alloc(&scratch_mc_header, scratch_size, mem_type);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to allocate scratch for allreduce ring");
        ucc_tl_ucp_put_task(task);
        return status;
    }

    task->reduce_scatter_ring.scratch           = scratch_mc_header->addr;
    task->reduce_scatter_ring.scratch_mc_header = scratch_mc_header;
    task->reduce_scatter_ring.max_block_count   = block_count;
    task->reduce_scatter_ring.s_scratch_busy[0] = 0;
    task->reduce_scatter_ring.s_scratch_busy[1] = 0;

    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post      = ucc_tl_ucp_allreduce_ring_start;
    task->super.progress  = ucc_tl_ucp_allreduce_ring_progress;
    task->super.finalize  = ucc_tl_ucp_allreduce_ring_finalize;

    *task_h = &task->super;
    return UCC_OK;
}
