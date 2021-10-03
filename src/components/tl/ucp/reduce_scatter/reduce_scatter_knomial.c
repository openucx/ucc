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

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_scatter_kn.phase = _phase;                                \
    } while (0)

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args  = &coll_task->args;
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;
    uint8_t                node_type = task->reduce_scatter_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->reduce_scatter_kn.p;
    void                  *scratch   = task->reduce_scatter_kn.scratch;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->dst.info.mem_type;
    size_t                 count     = args->dst.info.count;
    ucc_datatype_t         dt        = args->dst.info.datatype;
    void                  *sbuf      = UCC_IS_INPLACE(*args) ?
        rbuf : args->src.info.buffer;
    size_t                 dt_size   = ucc_dt_size(dt);
    size_t                 data_size = count * dt_size;
    ucc_rank_t             size      = team->size;
    ucc_rank_t             rank      = team->rank;
    ptrdiff_t              peer_seg_offset, local_seg_offset, offset;
    ucc_rank_t             peer, step_radix, peer_seg_index, local_seg_index;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;
    void                  *reduce_data, *local_data;

    local_seg_count = 0;
    block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
    UCC_KN_GOTO_PHASE(task->reduce_scatter_kn.phase);

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

UCC_KN_PHASE_EXTRA:
    if ((KN_NODE_PROXY == node_type) || (KN_NODE_EXTRA == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return task->super.super.status;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        } else {
            if (UCC_OK != (status = ucc_dt_reduce(sbuf, scratch, rbuf, count,
                                                  dt, mem_type, args))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
    }
    while (!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        sbuf        = (p->iteration == 0)
            ? ((KN_NODE_PROXY == node_type || UCC_IS_INPLACE(*args)) ?
               args->dst.info.buffer : args->src.info.buffer)
                          : task->reduce_scatter_kn.scratch;
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            peer_seg_index =
                ucc_sra_kn_compute_seg_index(peer, p->radix_pow, p);
            peer_seg_count = ucc_sra_kn_compute_seg_size(
                block_count, step_radix, peer_seg_index);
            peer_seg_offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, peer_seg_index);
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
        }

        local_seg_index = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
        local_seg_count = ucc_sra_kn_compute_seg_size(block_count, step_radix,
                                                      local_seg_index);

        rbuf = task->reduce_scatter_kn.scratch;
        if (p->iteration != 0) {
            rbuf = PTR_OFFSET(rbuf, block_count * dt_size);
        }
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(rbuf, local_seg_count * dt_size,
                                             mem_type, peer, team, task),
                          task, out);
            rbuf = PTR_OFFSET(rbuf, local_seg_count * dt_size);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return task->super.super.status;
        }
        if (task->send_posted > p->iteration * (radix - 1)) {
            sbuf       = (p->iteration == 0)
                ? ((KN_NODE_PROXY == node_type  || UCC_IS_INPLACE(*args)) ?
                   args->dst.info.buffer : args->src.info.buffer)
                             : task->reduce_scatter_kn.scratch;
            rbuf       = (p->iteration != 0)
                             ? PTR_OFFSET(task->reduce_scatter_kn.scratch,
                                    block_count * dt_size)
                             : task->reduce_scatter_kn.scratch;
            step_radix = ucc_sra_kn_compute_step_radix(rank, size, p);
            local_seg_index =
                ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
            local_seg_count = ucc_sra_kn_compute_seg_size(
                block_count, step_radix, local_seg_index);
            local_seg_offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, local_seg_index);
            local_data  = PTR_OFFSET(sbuf, local_seg_offset * dt_size);
            reduce_data = task->reduce_scatter_kn.scratch;
            if (UCC_OK != (status = ucc_dt_reduce_multi(
                               local_data, rbuf, reduce_data,
                               task->send_posted - p->iteration * (radix - 1),
                               local_seg_count, local_seg_count * dt_size, dt,
                               mem_type, args))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
        ucc_knomial_pattern_next_iteration(p);
    }

    offset = ucc_sra_kn_get_offset(count, dt_size, rank, size, radix);
    status = ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer, offset),
                           task->reduce_scatter_kn.scratch,
                           local_seg_count * dt_size, mem_type, mem_type);

    if (UCC_OK != status) {
        return status;
    }
UCC_KN_PHASE_PROXY: /* unused label */
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_done",
                                     0);
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args = &coll_task->args;
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;
    uint8_t            node_type;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_start",
                                     0);
    ucc_tl_ucp_task_reset(task);

    ucc_knomial_pattern_init(team->size, team->rank,
                             task->reduce_scatter_kn.p.radix,
                             &task->reduce_scatter_kn.p);
    node_type = task->reduce_scatter_kn.p.node_type;
    if (!(UCC_IS_INPLACE(*args) || (KN_NODE_PROXY == node_type))) {
        task->reduce_scatter_kn.scratch = args->dst.info.buffer;
    }
    task->reduce_scatter_kn.phase = UCC_KN_PHASE_INIT;

    status = ucc_tl_ucp_reduce_scatter_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    uint8_t            node_type = task->reduce_scatter_kn.p.node_type;

    if (UCC_IS_INPLACE(coll_task->args) || (KN_NODE_PROXY == node_type)) {
        ucc_mc_free(task->reduce_scatter_kn.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size      = tl_team->size;
    ucc_rank_t         rank      = tl_team->rank;
    size_t             count     = coll_args->args.dst.info.count;
    ucc_datatype_t     dt        = coll_args->args.dst.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    ucc_memory_type_t  mem_type  = coll_args->args.dst.info.mem_type;
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);
    ucc_knomial_pattern_init(size, rank, radix, &task->reduce_scatter_kn.p);

    if (UCC_IS_INPLACE(coll_args->args) ||
        (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type)) {
        status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch_mc_header,
                              data_size, mem_type);
        task->reduce_scatter_kn.scratch =
            task->reduce_scatter_kn.scratch_mc_header->addr;
        if (UCC_OK != status) {
            return status;
        }
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = tl_team->size;
    size_t             count   = coll_args->args.dst.info.count;
    ucc_kn_radix_t     radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatter_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix, size, count);
    return ucc_tl_ucp_reduce_scatter_knomial_init_r(coll_args, team, task_h,
                                                    radix);
}
