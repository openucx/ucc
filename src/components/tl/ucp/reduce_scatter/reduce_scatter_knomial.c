/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "core/ucc_mc.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

enum {
    PHASE_INIT,
    PHASE_LOOP,  /* main loop of recursive k-ing */
    PHASE_EXTRA /* recv from extra rank */
};

#define CHECK_PHASE(_p)                         \
    case _p:                                    \
    goto _p;                                    \
    break;

#define GOTO_PHASE(_phase)                                                     \
    do {                                                                       \
        switch (_phase) {                                                      \
            CHECK_PHASE(PHASE_EXTRA);                                          \
            CHECK_PHASE(PHASE_LOOP);                                           \
        case PHASE_INIT:                                                       \
            break;                                                             \
        };                                                                     \
    } while (0)


#define SAVE_STATE(_phase)                      \
    do {                                        \
        task->reduce_scatter_kn.phase = _phase;      \
    } while (0)


ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team       = task->team;
    ucc_kn_radix_t         radix      = task->reduce_scatter_kn.p.radix;
    uint8_t                node_type  = task->reduce_scatter_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->reduce_scatter_kn.p;
    void                  *scratch    = task->reduce_scatter_kn.scratch;
    void                  *sbuf       = task->args.src.info.buffer;
    void                  *rbuf       = task->args.dst.info.buffer;
    ucc_memory_type_t      mem_type   = task->args.src.info.mem_type;
    size_t                 count      = task->args.src.info.count;
    ucc_datatype_t         dt         = task->args.src.info.datatype;
    size_t                 dt_size    = ucc_dt_size(dt);
    size_t                 data_size  = count * dt_size;
    ucc_rank_t             size       = team->size;
    ucc_rank_t             rank       = team->rank;
    ptrdiff_t              peer_seg_offset, local_seg_offset, offset;
    ucc_rank_t             peer, step_radix, peer_seg_index, local_seg_index;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;
    void                  *reduce_data, *local_data;

    local_seg_count = 0;
    block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
    GOTO_PHASE(task->reduce_scatter_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(sbuf, data_size, mem_type, peer, team, task),
            task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(scratch, data_size, mem_type, peer, team, task),
            task, out);
    }

PHASE_EXTRA:
    if ((KN_NODE_PROXY == node_type) || (KN_NODE_EXTRA == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_EXTRA);
            return task->super.super.status;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        } else {
            if (UCC_OK != (status = ucc_dt_reduce(sbuf, scratch, rbuf,
                                                  count, dt, mem_type, &task->args))) {
                tl_error(UCC_TL_TEAM_LIB(task->team),
                         "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
            task->args.src.info.buffer = task->args.dst.info.buffer;
        }
    }
    while(!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        sbuf = (p->iteration == 0) ? task->args.src.info.buffer:
            task->reduce_scatter_kn.scratch;
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            peer_seg_index  = ucc_sra_kn_compute_seg_index(peer, p->radix_pow, p);
            peer_seg_count  = ucc_sra_kn_compute_seg_size(block_count, step_radix, peer_seg_index);
            peer_seg_offset = ucc_sra_kn_compute_seg_offset(block_count, step_radix, peer_seg_index);
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, peer_seg_offset*dt_size),
                                   peer_seg_count*dt_size, mem_type, peer, team, task),
                task, out);
        }

        local_seg_index  = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
        local_seg_count  = ucc_sra_kn_compute_seg_size(block_count, step_radix, local_seg_index);

        rbuf = task->reduce_scatter_kn.scratch;
        if (p->iteration != 0) {
            rbuf = PTR_OFFSET(rbuf, block_count*dt_size);
        }
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(rbuf, local_seg_count*dt_size, mem_type, peer, team, task),
                task, out);
            rbuf = PTR_OFFSET(rbuf, local_seg_count*dt_size);
        }
    PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_LOOP);
            return task->super.super.status;
        }
        if (task->send_posted > p->iteration * (radix - 1)) {
            sbuf = task->args.src.info.buffer;
            rbuf = task->reduce_scatter_kn.scratch;
            if (p->iteration != 0) {
                sbuf = task->reduce_scatter_kn.scratch;
                rbuf = PTR_OFFSET(rbuf, block_count*dt_size);
            }
            step_radix        = ucc_sra_kn_compute_step_radix(rank, size, p);
            local_seg_index   = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
            local_seg_count   = ucc_sra_kn_compute_seg_size(block_count, step_radix, local_seg_index);
            local_seg_offset  = ucc_sra_kn_compute_seg_offset(block_count, step_radix, local_seg_index);
            local_data        = PTR_OFFSET(sbuf, local_seg_offset*dt_size);
            reduce_data       = task->reduce_scatter_kn.scratch;
            if (UCC_OK != (status = ucc_dt_reduce_multi(
                               local_data, rbuf, reduce_data,
                               task->send_posted - p->iteration * (radix - 1), local_seg_count,
                               local_seg_count*dt_size,
                               dt, mem_type, &task->args))) {
                tl_error(UCC_TL_TEAM_LIB(task->team),
                         "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
        ucc_knomial_pattern_next_iteration(p);
    }

    offset = ucc_sra_kn_get_offset(count, dt_size, rank, size, radix);
    status = ucc_mc_memcpy(PTR_OFFSET(task->args.dst.info.buffer, offset),
                           task->reduce_scatter_kn.scratch, local_seg_count*dt_size, mem_type, mem_type);

    if (UCC_OK != status) {
        return status;
    }
out:
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    ucc_status_t       status;
    task->super.super.status = UCC_INPROGRESS;

    status = ucc_tl_ucp_reduce_scatter_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (status < 0) {
        return status;
    } else {
        ucc_event_manager_notify(coll_task, UCC_EVENT_COMPLETED);
        if (coll_task->flags & UCC_COLL_TASK_FLAG_INTERNAL) {
            coll_task->finalize(coll_task);
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    uint8_t            node_type  = task->reduce_scatter_kn.p.node_type;
    ucc_memory_type_t  mem_type   = task->args.src.info.mem_type;
    if (UCC_IS_INPLACE(task->args) ||
        (KN_NODE_PROXY == node_type)) {
        ucc_mc_free(task->reduce_scatter_kn.scratch, mem_type);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}


ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init_r(ucc_base_coll_args_t *coll_args,
                                                      ucc_base_team_t *team,
                                                      ucc_coll_task_t **task_h,
                                                      ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size      = tl_team->size;
    ucc_rank_t         rank      = tl_team->rank;
    size_t             count     = coll_args->args.src.info.count;
    ucc_datatype_t     dt        = coll_args->args.src.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    ucc_memory_type_t  mem_type  = coll_args->args.src.info.mem_type;
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    task->reduce_scatter_kn.phase   = PHASE_INIT;
    task->reduce_scatter_kn.scratch = task->args.dst.info.buffer;
    ucc_assert(task->args.src.info.mem_type ==
               task->args.dst.info.mem_type);
    ucc_knomial_pattern_init(size, rank, radix,
                             &task->reduce_scatter_kn.p);

    if (UCC_IS_INPLACE(task->args) ||
        (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type)) {
        status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch, data_size,
                              mem_type);
        if (UCC_OK != status) {
            return status;
        }
        if (UCC_IS_INPLACE(task->args)) {
            task->args.src.info.buffer = task->args.dst.info.buffer;
        }
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                                    ucc_base_team_t *team,
                                                    ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size      = tl_team->size;
    size_t             count     = coll_args->args.src.info.count;
    ucc_kn_radix_t     radix;

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->
                    cfg.reduce_scatter_kn_radix, size);
    if (((count + radix - 1)/radix*(radix-1) > count) ||
        ((radix - 1) > count)) {
        radix = 2;
    }
    return ucc_tl_ucp_reduce_scatter_knomial_init_r(coll_args, team, task_h, radix);
}
