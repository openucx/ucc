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
        task->allgather_kn.phase = _phase;                                     \
    } while (0)

ucc_status_t ucc_tl_ucp_allgather_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team  = task->team;
    ucc_kn_radix_t         radix = task->allgather_kn.p.radix;
    uint8_t                node_type  = task->allgather_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->allgather_kn.p;
    void                  *rbuf       = task->args.dst.info.buffer;
    ucc_memory_type_t      mem_type   = task->args.src.info.mem_type;
    size_t                 count      = task->args.src.info.count;
    ucc_datatype_t         dt         = task->args.src.info.datatype;
    size_t                 dt_size    = ucc_dt_size(dt);
    size_t                 data_size  = count * dt_size;
    ucc_rank_t             size       = team->size;
    ucc_rank_t             rank       = team->rank;
    void                  *sbuf;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer, step_radix, peer_seg_index, local_seg_index;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;

    UCC_KN_GOTO_PHASE(task->allgather_kn.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(rbuf, data_size, mem_type, peer, team, task),
            task, out);
    }
UCC_KN_PHASE_EXTRA:
    if (KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return task->super.super.status;
        }
        goto out;
    }
    while (!ucc_knomial_pattern_loop_done_backward(p)) {
        step_radix       = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count      = ucc_sra_kn_compute_block_count(count, rank, p);
        local_seg_index  = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
        local_seg_count  = ucc_sra_kn_compute_seg_size(block_count, step_radix,
                                                      local_seg_index);
        local_seg_offset = ucc_sra_kn_compute_seg_offset(
            block_count, step_radix, local_seg_index);

        sbuf = task->args.src.info.buffer;
        rbuf = PTR_OFFSET(sbuf, -local_seg_offset * dt_size);
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(sbuf, local_seg_count * dt_size,
                                             mem_type, peer, team, task),
                          task, out);
        }
        task->args.src.info.buffer = rbuf;

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
                ucc_tl_ucp_recv_nb(PTR_OFFSET(rbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return task->super.super.status;
        }
        ucc_knomial_pattern_next_iteration_backward(p);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->args.dst.info.buffer, data_size,
                                         mem_type, peer, team, task),
                      task, out);
    } else {
        goto out;
    }
UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return task->super.super.status;
    }

out:
    task->super.super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_done", 0);
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_allgather_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    ucc_rank_t         size = team->size;
    ucc_rank_t         rank = team->rank;
    ucc_status_t       status;
    ptrdiff_t          offset;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_start", 0);
    task->allgather_kn.phase = UCC_KN_PHASE_INIT;
    ucc_assert(task->args.src.info.mem_type == task->args.dst.info.mem_type);

    offset = ucc_sra_kn_get_offset(task->args.src.info.count,
                                   ucc_dt_size(task->args.src.info.datatype),
                                   rank, size, task->allgather_kn.p.radix);
    if (!UCC_IS_INPLACE(task->args)) {
        status = ucc_mc_memcpy(PTR_OFFSET(task->args.dst.info.buffer, offset),
                               task->args.src.info.buffer,
                               task->args.src.info.count *
                                   ucc_dt_size(task->args.src.info.datatype),
                               task->args.dst.info.mem_type,
                               task->args.src.info.mem_type);
        if (UCC_OK != status) {
            return status;
        }
    }
    task->args.src.info.buffer = PTR_OFFSET(task->args.dst.info.buffer, offset);
    task->super.super.status   = UCC_INPROGRESS;

    status = ucc_tl_ucp_allgather_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = tl_team->size;
    ucc_rank_t         rank    = tl_team->rank;
    ucc_tl_ucp_task_t *task;
    task = ucc_tl_ucp_init_task(coll_args, team);

    ucc_knomial_pattern_init_backward(size, rank, radix, &task->allgather_kn.p);

    task->super.post     = ucc_tl_ucp_allgather_knomial_start;
    task->super.progress = ucc_tl_ucp_allgather_knomial_progress;
    *task_h              = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = tl_team->size;
    ucc_kn_radix_t     radix;

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allgather_kn_radix, size);
    return ucc_tl_ucp_allgather_knomial_init_r(coll_args, team, task_h, radix);
}
