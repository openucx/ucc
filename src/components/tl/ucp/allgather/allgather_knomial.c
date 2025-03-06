/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "tl_ucp_copy.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allgather_kn.phase = _phase;                                     \
    } while (0)

#define GET_LOCAL_COUNT(_args, _size, _rank)                                   \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? ucc_coll_args_get_count((_args), (_args)->dst.info_v.counts,         \
                                  (_rank))                                     \
        : (_args)->dst.info.count / (_size)

#define GET_TOTAL_COUNT(_args, _size)                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? ucc_coll_args_get_total_count((_args), (_args)->dst.info_v.counts,   \
                                        (_size))                               \
        : (_args)->dst.info.count

#define GET_DT(_args)                                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? (_args)->dst.info_v.datatype                                         \
        : (_args)->dst.info.datatype

#define GET_DST(_args)                                                         \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? (_args)->dst.info_v.buffer                                           \
        : (_args)->dst.info.buffer

#define GET_MT(_args)                                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? (_args)->dst.info_v.mem_type                                         \
        : (_args)->dst.info.mem_type

/* Bcast will first call scatter and then allgather.
 * In case of non-full tree with "extra" ranks, scatter will give each rank
 * a new virtual rank number - "vrank".
 * As such allgather must keep to this ranking to be aligned with scatter.
 */

void ucc_tl_ucp_allgather_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task      = ucc_derived_of(coll_task,
                                                      ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team      = TASK_TEAM(task);
    ucc_tl_ucp_context_t  *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    ucc_kn_radix_t         radix     = task->allgather_kn.p.radix;
    uint8_t                node_type = task->allgather_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->allgather_kn.p;
    void                  *rbuf      = GET_DST(args);
    ucc_memory_type_t      mem_type  = GET_MT(args);
    size_t                 dt_size   = ucc_dt_size(GET_DT(args));
    ucc_rank_t             size      = task->subset.map.ep_num;
    size_t                 data_size = GET_TOTAL_COUNT(args, size);
    ucc_rank_t             broot     = args->coll_type == UCC_COLL_TYPE_BCAST ?
                                       args->root : 0;
    ucc_rank_t             rank      = VRANK(task->subset.myrank, broot, size);
    size_t                 local     = GET_LOCAL_COUNT(args, size, rank);
    void                  *sbuf;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer, peer_dist;
    ucc_kn_radix_t         loop_step;
    size_t                 peer_seg_count, local_seg_count;
    ucc_status_t           status;
    size_t                 extra_count;

    COPY_TASK_TEST(UCC_KN_PHASE_INIT, ctx, "failed during copy task test",
                   task->allgather_kn.copy_task);
    UCC_KN_GOTO_PHASE(task->allgather_kn.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        if (p->type != KN_PATTERN_ALLGATHERX) {
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->allgather_kn.sbuf,
                                             local * dt_size, mem_type,
                                             ucc_ep_map_eval(task->subset.map,
                                             INV_VRANK(peer,broot,size)),
                                             team, task),
                          task, out);
        }
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(rbuf, data_size * dt_size, mem_type,
                                         ucc_ep_map_eval(task->subset.map,
                                         INV_VRANK(peer,broot,size)),
                                         team, task),
                      task, out);
    }
    if ((p->type != KN_PATTERN_ALLGATHERX) && (node_type == KN_NODE_PROXY)) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        extra_count =
            coll_task->bargs.args.coll_type == UCC_COLL_TYPE_ALLGATHER
                ? local
                : ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        peer = ucc_ep_map_eval(task->subset.map, peer);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(PTR_OFFSET(task->allgather_kn.sbuf,
                                        local * dt_size), extra_count * dt_size,
                                        mem_type, peer, team, task),
                      task, out);
    }

UCC_KN_PHASE_EXTRA:
    if ((KN_NODE_EXTRA == node_type) || (KN_NODE_PROXY == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        }
    }
    while (!ucc_knomial_pattern_loop_done(p)) {
        ucc_kn_ag_pattern_peer_seg(rank, p, &local_seg_count,
                                   &local_seg_offset);
        sbuf = PTR_OFFSET(rbuf, local_seg_offset * dt_size);

        for (loop_step = radix - 1; loop_step > 0; loop_step--) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST) {
                peer_dist = ucc_knomial_calc_recv_dist(size - p->n_extra,
                        ucc_knomial_pattern_loop_rank(p, peer), p->radix, 0);
                if (peer_dist < task->allgather_kn.recv_dist) {
                    continue;
                }
            }
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(sbuf, local_seg_count * dt_size,
                                             mem_type,
                                             ucc_ep_map_eval(task->subset.map,
                                             INV_VRANK(peer, broot, size)),
                                             team, task),
                          task, out);
        }

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            ucc_kn_ag_pattern_peer_seg(peer, p, &peer_seg_count,
                                       &peer_seg_offset);

            if (coll_task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST) {
                peer_dist = ucc_knomial_calc_recv_dist(size - p->n_extra,
                        ucc_knomial_pattern_loop_rank(p, peer), p->radix, 0);
                if (peer_dist > task->allgather_kn.recv_dist) {
                    continue;
                }
            }
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(PTR_OFFSET(rbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type,
                                   ucc_ep_map_eval(task->subset.map,
                                   INV_VRANK(peer, broot, size)),
                                   team, task),
                task, out);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test_recv(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        ucc_kn_ag_pattern_next_iter(p);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(rbuf, data_size * dt_size, mem_type,
                                         ucc_ep_map_eval(task->subset.map,
                                         INV_VRANK(peer, broot, size)),
                                         team, task),
                      task, out);
    }
UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return;
    }

out:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t          *task  = ucc_derived_of(coll_task,
                                                       ucc_tl_ucp_task_t);
    ucc_coll_args_t            *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t          *team  = TASK_TEAM(task);
    ucc_tl_ucp_context_t       *ctx   = UCC_TL_UCP_TEAM_CTX(team);
    ucc_coll_type_t             ct    = args->coll_type;
    ucc_rank_t                  size  = task->subset.map.ep_num;
    ucc_kn_radix_t              radix = task->allgather_kn.p.radix;
    ucc_knomial_pattern_t      *p     = &task->allgather_kn.p;
    ucc_rank_t                  rank  = VRANK(task->subset.myrank,
                                              ct == UCC_COLL_TYPE_BCAST ?
                                              args->root : 0, size);
    ucc_status_t       status;
    ptrdiff_t          offset;
    void              *rbuf;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->allgather_kn.copy_task = NULL;
    task->allgather_kn.phase     = UCC_KN_PHASE_INIT;
    if (ct == UCC_COLL_TYPE_ALLGATHER) {
        ucc_kn_ag_pattern_init(size, rank, radix, args->dst.info.count,
                               &task->allgather_kn.p);
        offset = ucc_buffer_block_offset(args->dst.info.count, size, rank) *
                 ucc_dt_size(args->dst.info.datatype);
        rbuf   = args->dst.info.buffer;
        if (!UCC_IS_INPLACE(*args)) {
            status = ctx->copy.post(PTR_OFFSET(args->dst.info.buffer, offset),
                                               args->dst.info.mem_type,
                                               args->src.info.buffer,
                                               args->src.info.mem_type,
                                               args->src.info.count *
                                                  ucc_dt_size(args->src.info.datatype),
                                               task,
                                               &task->allgather_kn.copy_task);
            if (ucc_unlikely(status != UCC_OK)) {
                task->super.status = status;
                return status;
            }
        }
    } else if (ct == UCC_COLL_TYPE_ALLGATHERV) {
        ucc_kn_agv_pattern_init(size, rank, radix, args->dst.info_v.counts,
                                UCC_COLL_ARGS_COUNT64(args),
                                &task->allgather_kn.p);
        offset = ucc_buffer_vector_block_offset(args->dst.info_v.counts,
                                                UCC_COLL_ARGS_COUNT64(args),
                                                rank) *
                 ucc_dt_size(args->dst.info_v.datatype);
        rbuf   = args->dst.info_v.buffer;
        if (!UCC_IS_INPLACE(*args)) {
            status = ctx->copy.post(PTR_OFFSET(args->dst.info_v.buffer, offset),
                                    args->dst.info_v.mem_type,
                                    args->src.info.buffer,
                                    args->src.info.mem_type,
                                    args->src.info.count *
                                      ucc_dt_size(args->src.info.datatype),
                                    task,
                                    &task->allgather_kn.copy_task);
            if (ucc_unlikely(status != UCC_OK)) {
                task->super.status = status;
                return status;
            }
        }
    } else {
        ucc_kn_agx_pattern_init(size, rank, radix, args->dst.info.count,
                                &task->allgather_kn.p);
        offset = ucc_sra_kn_get_offset(args->dst.info.count,
                                    ucc_dt_size(args->dst.info.datatype), rank,
                                    size, radix);
        rbuf   = args->dst.info.buffer;
        task->allgather_kn.recv_dist = ucc_knomial_calc_recv_dist(
            size - p->n_extra,
            ucc_knomial_pattern_loop_rank(p, rank),
            p->radix, 0);
    }
    task->allgather_kn.sbuf = PTR_OFFSET(rbuf, offset);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_task_t *task;
    ucc_sbgp_t        *sbgp;

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (tl_team->cfg.use_reordering &&
        coll_args->args.coll_type == UCC_COLL_TYPE_ALLREDUCE) {
        sbgp = ucc_topo_get_sbgp(tl_team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }
    task->allgather_kn.p.radix = radix;
    if (!UCC_IS_INPLACE(coll_args->args)) {
        if (ctx->cfg.local_copy_type == UCC_TL_UCP_LOCAL_COPY_TYPE_EC) {
            task->super.flags         |= UCC_COLL_TASK_FLAG_EXECUTOR;
        }
    }
    task->super.post           = ucc_tl_ucp_allgather_knomial_start;
    task->super.progress       = ucc_tl_ucp_allgather_knomial_progress;
    *task_h                    = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_mrange_uint_t *p       = &tl_team->cfg.allgather_kn_radix;
    ucc_rank_t         tsize   = UCC_TL_TEAM_SIZE(tl_team);
    ucc_memory_type_t  mtype   = GET_MT(&coll_args->args);
    size_t             count   = GET_TOTAL_COUNT(&coll_args->args, tsize);
    ucc_datatype_t     dtype   = GET_DT(&coll_args->args);
    ucc_kn_radix_t     radix;

    radix = ucc_tl_ucp_get_knomial_radix(tl_team, count, dtype, mtype, p, 0);

    return ucc_tl_ucp_allgather_knomial_init_r(coll_args, team, task_h, radix);
}
