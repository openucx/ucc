/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allgather_kn.phase = _phase;                                     \
    } while (0)

void ucc_tl_ucp_allgather_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->allgather_kn.p.radix;
    uint8_t                node_type  = task->allgather_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->allgather_kn.p;
    void                  *rbuf       = args->dst.info.buffer;
    ucc_memory_type_t      mem_type   = args->dst.info.mem_type;
    size_t                 count      = args->dst.info.count;
    ucc_datatype_t         dt         = args->dst.info.datatype;
    size_t                 dt_size    = ucc_dt_size(dt);
    size_t                 data_size  = count * dt_size;
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t             broot      = 0;
    void                  *sbuf;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer, step_radix, peer_seg_index, local_seg_index;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;
    ucc_status_t           status;

    if (task->allgather_kn.etask != NULL) {
        status = ucc_ee_executor_task_test(task->allgather_kn.etask);
        if (status == UCC_INPROGRESS) {
            task->super.status = status;
            return;
        }
        ucc_ee_executor_task_finalize(task->allgather_kn.etask);
        task->allgather_kn.etask = NULL;
        if (ucc_unlikely(status < 0)) {
            task->super.status = status;
            return;
        }
    }
    /* Bcast will first call scatter and then allgather.
       In case of non-full tree with "extra" ranks, scatter will give each rank
       a new virtual rank number - "vrank".
       As such allgather must keep to this ranking to be aligned with scatter.
    */
    if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST) {
        broot = coll_task->bargs.args.root;
        rank = VRANK(rank, broot, size);
    }

    UCC_KN_GOTO_PHASE(task->allgather_kn.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(rbuf, data_size, mem_type,
                               INV_VRANK(peer, broot, size), team, task),
            task, out);
    }
UCC_KN_PHASE_EXTRA:
    if (KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return;
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

        sbuf = task->allgather_kn.sbuf;
        rbuf = PTR_OFFSET(sbuf, -local_seg_offset * dt_size);
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(sbuf, local_seg_count * dt_size,
                                             mem_type,
                                             INV_VRANK(peer, broot, size), team, task),
                          task, out);
        }
        task->allgather_kn.sbuf = rbuf;

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
                                   peer_seg_count * dt_size, mem_type,
                                   INV_VRANK(peer, broot, size),
                                   team, task),
                task, out);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        ucc_knomial_pattern_next_iteration_backward(p);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(args->dst.info.buffer, data_size,
                                         mem_type,
                                         INV_VRANK(peer, broot, size), team, task),
                      task, out);
    } else {
        goto out;
    }
UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return;
    }

out:
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         rank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size  = UCC_TL_TEAM_SIZE(team);
    ucc_kn_radix_t     radix = task->allgather_kn.p.radix;
    ucc_rank_t         broot = 0;
    ucc_status_t       status;
    ptrdiff_t          offset;
    ucc_ee_executor_task_args_t eargs;
    ucc_ee_executor_t *exec;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST) {
        broot = coll_task->bargs.args.root;
        rank = VRANK(rank, broot, size);
    }

    task->allgather_kn.etask = NULL;
    task->allgather_kn.phase = UCC_KN_PHASE_INIT;
    ucc_assert(args->src.info.mem_type == args->dst.info.mem_type);

    ucc_knomial_pattern_init_backward(size, rank, radix, &task->allgather_kn.p);
    offset = ucc_sra_kn_get_offset(args->dst.info.count,
                                   ucc_dt_size(args->dst.info.datatype), rank,
                                   size, radix);
    if (!UCC_IS_INPLACE(*args)) {
        status = ucc_coll_task_get_executor(&task->super, &exec);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return status;
        }
        eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
        eargs.copy.dst  = PTR_OFFSET(args->dst.info.buffer, offset);
        eargs.copy.src  = args->src.info.buffer;
        eargs.copy.len  =
            args->src.info.count * ucc_dt_size(args->src.info.datatype);
        status = ucc_ee_executor_task_post(exec, &eargs,
                                           &task->allgather_kn.etask);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return status;
        }
    }
    task->allgather_kn.sbuf = PTR_OFFSET(args->dst.info.buffer, offset);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         rank    = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_rank_t         broot   = 0;
    ucc_tl_ucp_task_t *task;

    if (coll_args->args.coll_type == UCC_COLL_TYPE_BCAST) {
        broot = coll_args->args.root;
        rank = VRANK(rank, broot, size);
    }
    task = ucc_tl_ucp_init_task(coll_args, team);
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_allgather_knomial_start;
    task->super.progress = ucc_tl_ucp_allgather_knomial_progress;
    ucc_knomial_pattern_init_backward(size, rank, radix, &task->allgather_kn.p);

    *task_h              = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_kn_radix_t     radix;

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allgather_kn_radix, size);
    return ucc_tl_ucp_allgather_knomial_init_r(coll_args, team, task_h, radix);
}
