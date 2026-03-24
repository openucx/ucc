/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "reduce_scatter.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_dt_reduce.h"
#include "coll_patterns/ring.h"

#define MAX_RINGS 8

static inline size_t
rs_ring_total_count(ucc_coll_args_t *args)
{
    return UCC_IS_INPLACE(*args) ? args->dst.info.count
                                : args->src.info.count;
}


void ucc_tl_ucp_reduce_scatter_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t    *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t  *team      = TASK_TEAM(task);
    ucc_ring_pattern_t *ring      = team->cuda_ring;
    ucc_rank_t          nrings    = ucc_min(MAX_RINGS, ring->num_rings);
    ucc_rank_t          tsize     = ucc_ring_pattern_size(ring, 0);
    size_t              total_cnt = rs_ring_total_count(args);
    size_t              block_cnt = total_cnt / tsize;
    void               *sbuf      = UCC_IS_INPLACE(*args)
                                        ? args->dst.info.buffer
                                        : args->src.info.buffer;
    void               *dst       = args->dst.info.buffer;
    void               *scratch   = task->reduce_scatter_ring.scratch;
    ucc_memory_type_t   mem_type  = args->dst.info.mem_type;
    ucc_datatype_t      dt        = args->dst.info.datatype;
    size_t              dt_size   = ucc_dt_size(dt);
    ucc_rank_t          rrank, adj_rrank, recv_block, send_block;
    ucc_rank_t          sendto, recvfrom, ring_id, step;
    size_t              ring_offset, ring_count, data_displ, data_size;
    ucc_status_t        status;
    int                 is_avg;
    void               *reduce_target;

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
            ring_offset = ucc_buffer_block_offset(block_cnt, nrings, ring_id);
            ring_count  = ucc_buffer_block_count(block_cnt, nrings, ring_id);

            if (step == (ucc_rank_t)(tsize - 2)) {
                if (UCC_IS_INPLACE(*args)) {
                    reduce_target = PTR_OFFSET(
                        dst,
                        (recv_block * block_cnt + ring_offset) * dt_size);
                } else {
                    reduce_target = PTR_OFFSET(dst, ring_offset * dt_size);
                }
            } else {
                reduce_target = PTR_OFFSET(
                    sbuf,
                    (recv_block * block_cnt + ring_offset) * dt_size);
            }

            status = ucc_dt_reduce(
                PTR_OFFSET(scratch, ring_offset * dt_size),
                PTR_OFFSET(sbuf,
                           (recv_block * block_cnt + ring_offset) * dt_size),
                reduce_target, ring_count, dt, args,
                is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                AVG_ALPHA(task), task->reduce_scatter_ring.executor,
                &task->reduce_scatter_ring.etask);
            if (UCC_OK != status) {
                tl_error(UCC_TASK_LIB(task),
                         "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
            EXEC_TASK_WAIT(task->reduce_scatter_ring.etask);
        }

        if (step + 1 >= (ucc_rank_t)(tsize - 1)) {
            task->super.status = UCC_OK;
            return;
        }

        step++;
        for (ring_id = 0; ring_id < nrings; ring_id++) {
            rrank       = ucc_ring_pattern_rank(ring, ring_id);
            sendto      = ucc_ring_pattern_get_send_peer(ring, ring_id, rrank);
            recvfrom    = ucc_ring_pattern_get_recv_peer(ring, ring_id, rrank);
            ring_offset = ucc_buffer_block_offset(block_cnt, nrings, ring_id);
            ring_count  = ucc_buffer_block_count(block_cnt, nrings, ring_id);

            adj_rrank   = (rrank + tsize - 1) % tsize;
            send_block  = ucc_ring_pattern_get_recv_block(ring, ring_id,
                                                          adj_rrank,
                                                          step - 1);
            data_displ = (send_block * block_cnt + ring_offset) * dt_size;
            data_size  = ring_count * dt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(
                PTR_OFFSET(sbuf, data_displ), data_size, mem_type,
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

    task->super.status = UCC_OK;
out:
    return;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ucc_ring_pattern_t *ring     = team->cuda_ring;
    ucc_rank_t          nrings   = ucc_min(MAX_RINGS, ring->num_rings);
    ucc_rank_t          tsize    = ucc_ring_pattern_size(ring, 0);
    size_t              total_cnt = rs_ring_total_count(args);
    size_t              block_cnt = total_cnt / tsize;
    ucc_datatype_t      dt       = args->dst.info.datatype;
    size_t              dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t   mem_type = args->dst.info.mem_type;
    void               *sbuf     = UCC_IS_INPLACE(*args)
                                       ? args->dst.info.buffer
                                       : args->src.info.buffer;
    void               *scratch  = task->reduce_scatter_ring.scratch;
    ucc_rank_t          ring_id, rrank, adj_rrank, send_block, sendto, recvfrom;
    size_t              ring_offset, ring_count, data_displ, data_size;
    ucc_status_t        status;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    status = ucc_coll_task_get_executor(&task->super,
                                        &task->reduce_scatter_ring.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

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
            PTR_OFFSET(sbuf, data_displ), data_size, mem_type, sendto,
            team, task), task, err);

        data_displ = ring_offset * dt_size;
        data_size  = ring_count * dt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(
            PTR_OFFSET(scratch, data_displ), data_size, mem_type, recvfrom,
            team, task), task, err);
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
err:
    return task->super.status;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    ucc_mc_free(task->reduce_scatter_ring.scratch_mc_header);
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_reduce_scatter_ring_init_common(
    ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_coll_args_t        *args = &TASK_ARGS(task);
    ucc_rank_t              tsize;
    size_t                  total_count, block_count, scratch_size;
    ucc_datatype_t          dt;
    size_t                  dt_size;
    ucc_memory_type_t       mem_type;
    ucc_mc_buffer_header_t *scratch_mc_header;
    ucc_status_t            status;

    if (!team->cuda_ring) {
        return UCC_ERR_NOT_FOUND;
    }

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op &&
        args->op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_IS_PERSISTENT(*args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    tsize       = ucc_ring_pattern_size(team->cuda_ring, 0);
    total_count = rs_ring_total_count(args);
    if (total_count % tsize != 0) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    block_count  = total_count / tsize;
    dt           = args->dst.info.datatype;
    dt_size      = ucc_dt_size(dt);
    mem_type     = args->dst.info.mem_type;
    scratch_size = block_count * dt_size;

    status = ucc_mc_alloc(&scratch_mc_header, scratch_size, mem_type);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to allocate scratch for reduce_scatter ring");
        return status;
    }

    task->reduce_scatter_ring.scratch           = scratch_mc_header->addr;
    task->reduce_scatter_ring.scratch_mc_header = scratch_mc_header;
    task->reduce_scatter_ring.max_block_count   = block_count;
    task->reduce_scatter_ring.s_scratch_busy[0] = 0;
    task->reduce_scatter_ring.s_scratch_busy[1] = 0;

    task->super.flags   |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_reduce_scatter_ring_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_ring_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_ring_finalize;

    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t      *team,
                                    ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_reduce_scatter_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
