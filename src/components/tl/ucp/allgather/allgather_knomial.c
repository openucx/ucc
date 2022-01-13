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
#include "allgather.h"
#include "allgatherv/allgatherv.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allgather_kn.phase = _phase;                                     \
    } while (0)

#define GET_TOTAL_COUNT(_args, _size)                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? ucc_coll_args_get_total_count((_args), (_args)->dst.info_v.counts,   \
                                        (_size))                               \
        : (_args)->dst.info.count

#define GET_LOCAL_COUNT(_args, _size, _rank)                                   \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? ucc_coll_args_get_count((_args), (_args)->dst.info_v.counts,         \
                                  (_rank))                                     \
        : (_args)->dst.info.count / (_size)

#define GET_DT(_args)                                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? (_args)->dst.info_v.datatype                                         \
        : (_args)->dst.info.datatype

#define GET_MT(_args)                                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? (_args)->dst.info_v.mem_type                                         \
        : (_args)->dst.info.mem_type

#define GET_DST(_args)                                                         \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? (_args)->dst.info_v.buffer                                           \
        : (_args)->dst.info.buffer

ucc_status_t ucc_tl_ucp_allgather_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->allgather_kn.p.radix;
    uint8_t                node_type  = task->allgather_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->allgather_kn.p;
    ucc_memory_type_t      mem_type   = GET_MT(args);
    ucc_datatype_t         datatype   = GET_DT(args);
    size_t                 dt_size    = ucc_dt_size(datatype);
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    size_t                 total      = GET_TOTAL_COUNT(args, size);
    size_t                 local      = GET_LOCAL_COUNT(args, size, rank);
    void *                 dst        = GET_DST(args);
    ucc_rank_t             broot      = 0;
    void                  *sbuf;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer;
    ucc_kn_radix_t         loop_step;
    size_t                 peer_seg_count, local_seg_count, extra_count;

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
        if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_ALLGATHER ||
            args->coll_type == UCC_COLL_TYPE_ALLGATHERV) {
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->allgather_kn.sbuf,
                                             local * dt_size, mem_type,
                                             INV_VRANK(peer, broot, size), team,
                                             task),
                          task, out);
        }
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(dst, total * dt_size, mem_type,
                                         INV_VRANK(peer, broot, size), team,
                                         task),
                      task, out);
    }
    if (KN_NODE_PROXY == node_type && p->type != KN_PATTERN_ALLGATHERX) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        extra_count =
            coll_task->bargs.args.coll_type == UCC_COLL_TYPE_ALLGATHER
                ? local
                : ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(
                          PTR_OFFSET(task->allgather_kn.sbuf, local * dt_size),
                          extra_count * dt_size, mem_type, peer, team, task),
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
        }
    }

    while (!ucc_knomial_pattern_loop_done(p)) {
        ucc_kn_ag_pattern_peer_seg(rank, p, &local_seg_count,
                                   &local_seg_offset);
        sbuf = PTR_OFFSET(dst, local_seg_offset * dt_size);

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(sbuf, local_seg_count * dt_size,
                                             mem_type,
                                             INV_VRANK(peer, broot, size), team, task),
                          task, out);
        }

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            ucc_kn_ag_pattern_peer_seg(peer, p, &peer_seg_count,
                                       &peer_seg_offset);

            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(PTR_OFFSET(dst, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type,
                                   INV_VRANK(peer, broot, size), team, task),
                task, out);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return task->super.super.status;
        }
        ucc_kn_ag_pattern_next_iter(p);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(dst, total * dt_size, mem_type,
                                         INV_VRANK(peer, broot, size), team,
                                         task),
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
    ucc_tl_ucp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args    = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team    = TASK_TEAM(task);
    ucc_rank_t         rank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(team);
    ucc_kn_radix_t     radix   = task->allgather_kn.p.radix;
    size_t             dt_size = ucc_dt_size(GET_DT(args));
    ucc_rank_t         broot   = 0;
    size_t             count;
    ucc_status_t       status;
    ptrdiff_t          offset;
    void *             start;
    ucc_memory_type_t  mt;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_start", 0);
    ucc_tl_ucp_task_reset(task);
    if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST) {
        broot = coll_task->bargs.args.root;
        rank = VRANK(rank, broot, size);
    }

    task->allgather_kn.phase = UCC_KN_PHASE_INIT;

    if (args->coll_type == UCC_COLL_TYPE_ALLGATHER ||
        args->coll_type == UCC_COLL_TYPE_ALLGATHERV) {
        mt    = GET_MT(args);
        count = GET_LOCAL_COUNT(args, size, rank);
        if (args->coll_type == UCC_COLL_TYPE_ALLGATHER) {
            ucc_kn_ag_pattern_init(size, rank, radix, args->dst.info.count,
                                   &task->allgather_kn.p);
            offset = ucc_buffer_block_offset(args->dst.info.count, size, rank) *
                     dt_size;
            start = PTR_OFFSET(args->dst.info.buffer, offset);
        } else {
            ucc_kn_agv_pattern_init(size, rank, radix, args->dst.info_v.counts,
                                    UCC_COLL_ARGS64(args),
                                    &task->allgather_kn.p);
            offset = vector_block_offset(&task->allgather_kn.p, rank);
            start  = PTR_OFFSET(args->dst.info_v.buffer, offset * dt_size);
        }
        if (!UCC_IS_INPLACE(*args)) {
            status =
                ucc_mc_memcpy(start, args->src.info.buffer, count * dt_size, mt,
                              args->src.info.mem_type);
            if (UCC_OK != status) {
                return status;
            }
        }
    } else {
        count = args->dst.info.count;
        start = NULL;
        ucc_kn_agx_pattern_init(size, rank, radix, count,
                                &task->allgather_kn.p);
    }
    task->allgather_kn.sbuf = start;

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
    ucc_rank_t         rank    = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_rank_t         broot   = 0;
    ucc_tl_ucp_task_t *task;
    ucc_knomial_pattern_t p;

    if (coll_args->args.coll_type == UCC_COLL_TYPE_ALLGATHER) {
        ucc_kn_ag_pattern_init(size, rank, radix,
                               coll_args->args.dst.info.count, &p);
    } else if (coll_args->args.coll_type == UCC_COLL_TYPE_ALLGATHERV) {
        if (!UCC_IS_DST_CONTIG(coll_args->args)) {
            return ucc_tl_ucp_allgatherv_ring_init(coll_args, team, task_h);
        }
        ucc_kn_agv_pattern_init(size, rank, radix,
                                coll_args->args.dst.info_v.counts,
                                UCC_COLL_ARGS64(&coll_args->args), &p);
    } else {
        if (coll_args->args.coll_type == UCC_COLL_TYPE_BCAST) {
            broot = coll_args->args.root;
            rank  = VRANK(rank, broot, size);
        }
        ucc_kn_agx_pattern_init(size, rank, radix,
                                coll_args->args.dst.info.count, &p);
    }

    task = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_allgather_knomial_start;
    task->super.progress = ucc_tl_ucp_allgather_knomial_progress;
    task->allgather_kn.p = p;
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
