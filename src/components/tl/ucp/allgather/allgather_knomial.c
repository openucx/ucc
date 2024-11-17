/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <stdio.h>

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allgather_kn.phase = _phase;                                     \
    } while (0)

#define GET_LOCAL_COUNT(_args, _size, _rank)                                   \
    ((_args)->coll_type == UCC_COLL_TYPE_ALLGATHERV)                           \
        ? ucc_coll_args_get_count((_args), (_args)->dst.info_v.counts,         \
                                  (_rank))                                     \
        : (_args)->dst.info.count / (_size)

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
    ucc_kn_radix_t         radix     = task->allgather_kn.p.radix;
    uint8_t                node_type = task->allgather_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->allgather_kn.p;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->dst.info.mem_type;
    size_t                 count     = args->dst.info.count;
    size_t                 dt_size   = ucc_dt_size(args->dst.info.datatype);
    size_t                 data_size = count * dt_size;
    ucc_rank_t             size      = task->subset.map.ep_num;
    ucc_rank_t             broot     = args->coll_type == UCC_COLL_TYPE_BCAST ?
                                       args->root : 0;
    ucc_rank_t             rank      = VRANK(task->subset.myrank, broot, size);
    size_t                 local     = GET_LOCAL_COUNT(args, size, rank);
    ucp_mem_h             *mh_list   = task->allgather_kn.mh_list;
    void                  *sbuf;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer, peer_dist;
    ucc_kn_radix_t         loop_step;
    size_t                 peer_seg_count, local_seg_count;
    ucc_status_t           status;
    size_t                 extra_count;

    EXEC_TASK_TEST(UCC_KN_PHASE_INIT, "failed during ee task test",
                   task->allgather_kn.etask);
    task->allgather_kn.etask = NULL;
    UCC_KN_GOTO_PHASE(task->allgather_kn.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        if (p->type != KN_PATTERN_ALLGATHERX) {
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb_with_mem(task->allgather_kn.sbuf,
                                             local * dt_size, mem_type,
                                             ucc_ep_map_eval(task->subset.map,
                                             INV_VRANK(peer,broot,size)),
                                             team, task, mh_list[task->allgather_kn.count_mh++]),
                          task, out);
            ucc_assert(task->allgather_kn.count_mh-1 <= task->allgather_kn.max_mh);
            
        }
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb_with_mem(rbuf, data_size, mem_type,
                                         ucc_ep_map_eval(task->subset.map,
                                         INV_VRANK(peer,broot,size)),
                                         team, task, mh_list[task->allgather_kn.count_mh++]),
                      task, out);
        ucc_assert(task->allgather_kn.count_mh-1 <= task->allgather_kn.max_mh);
    }
    if ((p->type != KN_PATTERN_ALLGATHERX) && (node_type == KN_NODE_PROXY)) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        extra_count = GET_LOCAL_COUNT(args, size, peer);
        peer = ucc_ep_map_eval(task->subset.map, peer);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb_with_mem(PTR_OFFSET(task->allgather_kn.sbuf,
                                        local * dt_size), extra_count * dt_size,
                                        mem_type, peer, team, task, mh_list[task->allgather_kn.count_mh++]),
                      task, out);
        ucc_assert(task->allgather_kn.count_mh-1 <= task->allgather_kn.max_mh);
    }

UCC_KN_PHASE_EXTRA:
    if ((KN_NODE_EXTRA == node_type) || (KN_NODE_PROXY == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test_with_etasks(task)) {
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
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb_with_mem(sbuf, local_seg_count * dt_size,
                                             mem_type,
                                             ucc_ep_map_eval(task->subset.map,
                                             INV_VRANK(peer, broot, size)),
                                             team, task, mh_list[task->allgather_kn.count_mh++]),
                          task, out);
            ucc_assert(task->allgather_kn.count_mh-1 <= task->allgather_kn.max_mh);
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
                ucc_tl_ucp_recv_nb_with_mem(PTR_OFFSET(rbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type,
                                   ucc_ep_map_eval(task->subset.map,
                                   INV_VRANK(peer, broot, size)),
                                   team, task, mh_list[task->allgather_kn.count_mh++]),
                task, out);
            ucc_assert(task->allgather_kn.count_mh-1 <= task->allgather_kn.max_mh);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test_recv_with_etasks(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        ucc_kn_ag_pattern_next_iter(p);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb_with_mem(args->dst.info.buffer, data_size,
                                         mem_type,
                                         ucc_ep_map_eval(task->subset.map,
                                         INV_VRANK(peer, broot, size)),
                                         team, task, mh_list[task->allgather_kn.count_mh++]),
                      task, out);
        ucc_assert(task->allgather_kn.count_mh-1 <= task->allgather_kn.max_mh);
    }
UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test_with_etasks(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return;
    }

out:
    ucc_assert(task->allgather_kn.count_mh-1 == task->allgather_kn.max_mh);
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
    ucc_coll_type_t             ct    = args->coll_type;
    ucc_rank_t                  size  = task->subset.map.ep_num;
    ucc_kn_radix_t              radix = task->allgather_kn.p.radix;
    ucc_knomial_pattern_t      *p     = &task->allgather_kn.p;
    ucc_rank_t                  rank  = VRANK(task->subset.myrank,
                                              ct == UCC_COLL_TYPE_BCAST ?
                                              args->root : 0, size);
    ucc_ee_executor_task_args_t eargs = {0};
    ucc_status_t       status;
    ptrdiff_t          offset;
    ucc_ee_executor_t *exec;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->allgather_kn.etask = NULL;
    task->allgather_kn.phase = UCC_KN_PHASE_INIT;
    if (ct == UCC_COLL_TYPE_ALLGATHER) {
        ucc_kn_ag_pattern_init(size, rank, radix, args->dst.info.count,
                               &task->allgather_kn.p);
        offset = ucc_buffer_block_offset(args->dst.info.count, size, rank) *
                 ucc_dt_size(args->dst.info.datatype);
        if (!UCC_IS_INPLACE(*args)) {
            status = ucc_coll_task_get_executor(&task->super, &exec);
            if (ucc_unlikely(status != UCC_OK)) {
                task->super.status = status;
                return status;
            }
            eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
            eargs.copy.dst  = PTR_OFFSET(args->dst.info.buffer, offset);
            eargs.copy.src  = args->src.info.buffer;
            eargs.copy.len  = args->src.info.count *
                              ucc_dt_size(args->src.info.datatype);
            status = ucc_ee_executor_task_post(exec, &eargs,
                                               &task->allgather_kn.etask);
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
        task->allgather_kn.recv_dist = ucc_knomial_calc_recv_dist(
            size - p->n_extra,
            ucc_knomial_pattern_loop_rank(p, rank),
            p->radix, 0);
    }
    task->allgather_kn.sbuf = PTR_OFFSET(args->dst.info.buffer, offset);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t register_memory(ucc_coll_task_t *coll_task){

    ucc_tl_ucp_task_t     *task      = ucc_derived_of(coll_task,
                                                      ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team      = TASK_TEAM(task);
    ucc_coll_type_t        ct        = args->coll_type;
    ucc_kn_radix_t         radix     = task->allgather_kn.p.radix;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->dst.info.mem_type;
    size_t                 count     = args->dst.info.count;
    size_t                 dt_size   = ucc_dt_size(args->dst.info.datatype);
    size_t                 data_size = count * dt_size;
    ucc_rank_t             size      = task->subset.map.ep_num;
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

    ucc_tl_ucp_context_t  *ctx = UCC_TL_UCP_TEAM_CTX(team);
    ucp_mem_map_params_t   mmap_params;
    int                    size_of_list = 1;
    int                    count_mh = 0;
    ucp_mem_h             *mh_list = (ucp_mem_h *)malloc(size_of_list * sizeof(ucp_mem_h));

    ptrdiff_t          offset;

    if (ct == UCC_COLL_TYPE_ALLGATHER) {
        ucc_kn_ag_pattern_init(size, rank, radix, args->dst.info.count,
                               &task->allgather_kn.p);
    } else {
        ucc_kn_agx_pattern_init(size, rank, radix, args->dst.info.count,
                                &task->allgather_kn.p);
    }

    offset = ucc_sra_kn_get_offset(count,
                                dt_size, rank,
                                size, radix);
    task->allgather_kn.sbuf = PTR_OFFSET(args->dst.info.buffer, offset);

    ucc_knomial_pattern_t *p         = &task->allgather_kn.p;
    uint8_t                node_type = task->allgather_kn.p.node_type;

    mmap_params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                            UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.memory_type = ucc_memtype_to_ucs[mem_type];
    if (KN_NODE_EXTRA == node_type) {
        if (p->type != KN_PATTERN_ALLGATHERX) {
            mmap_params.address     = task->allgather_kn.sbuf;
            mmap_params.length      = local * dt_size;
            MEM_MAP();
        }
        
        mmap_params.address     = rbuf;
        mmap_params.length      = data_size;
        MEM_MAP();
    }
    if ((p->type != KN_PATTERN_ALLGATHERX) && (node_type == KN_NODE_PROXY)) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        extra_count = GET_LOCAL_COUNT(args, size, peer);
        peer = ucc_ep_map_eval(task->subset.map, peer);
        mmap_params.address     = PTR_OFFSET(task->allgather_kn.sbuf,
                                        local * dt_size);
        mmap_params.length      = extra_count * dt_size;
        MEM_MAP();
    }

    if (KN_NODE_EXTRA == node_type) {
        goto out;
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
            mmap_params.address     = sbuf;
            mmap_params.length      = local_seg_count * dt_size;
            MEM_MAP();
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
            mmap_params.address     = PTR_OFFSET(rbuf, peer_seg_offset * dt_size);
            mmap_params.length      = peer_seg_count * dt_size;
            MEM_MAP();
        }
        ucc_kn_ag_pattern_next_iter(p);
    }

    if (KN_NODE_PROXY == node_type) {
        mmap_params.address     = args->dst.info.buffer;
        mmap_params.length      = data_size;
        MEM_MAP();
    }

out:
    task->allgather_kn.mh_list = mh_list;
    task->allgather_kn.max_mh = count_mh-1;
    task->allgather_kn.count_mh = 0;
    return UCC_OK;
}

ucc_status_t  ucc_tl_ucp_allgather_knomial_finalize(ucc_coll_task_t *coll_task){
    ucc_tl_ucp_task_t     *task      = ucc_derived_of(coll_task,
                                                      ucc_tl_ucp_task_t);
    ucc_status_t       status;
    ucc_tl_ucp_team_t     *team      = TASK_TEAM(task);
    ucc_tl_ucp_context_t  *ctx = UCC_TL_UCP_TEAM_CTX(team);
                                                      
    ucc_mpool_cleanup(&task->allgather_kn.etask_node_mpool, 1);
    for (int i=0; i<task->allgather_kn.max_mh+1; i++){
        ucp_mem_unmap(ctx->worker.ucp_context, task->allgather_kn.mh_list[i]);
    }
    free(task->allgather_kn.mh_list);
    status = ucc_tl_ucp_coll_finalize(&task->super);
    if (status < 0){
        tl_error(UCC_TASK_LIB(task),
                 "failed to initialize ucc_mpool");
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_sbgp_t        *sbgp;
    ucc_status_t       status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_mpool_init(&task->allgather_kn.etask_node_mpool, 0, sizeof(node_ucc_ee_executor_task_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX, NULL,
                            tl_team->super.super.context->ucc_context->thread_mode, "etasks_linked_list_nodes");
    if (status < 0){
        tl_error(UCC_TASK_LIB(task),
                 "failed to initialize ucc_mpool");
    }
    
    if (tl_team->cfg.use_reordering &&
        coll_args->args.coll_type == UCC_COLL_TYPE_ALLREDUCE) {
        sbgp = ucc_topo_get_sbgp(tl_team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }
    task->allgather_kn.etask_linked_list_head = NULL;
    task->allgather_kn.p.radix = radix;
    task->super.flags         |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_ucp_allgather_knomial_start;
    task->super.progress       = ucc_tl_ucp_allgather_knomial_progress;
    task->super.finalize       = ucc_tl_ucp_allgather_knomial_finalize;
    status = register_memory(&task->super);
    if (status < 0){
        tl_error(UCC_TASK_LIB(task),
                 "failed to register memory");
    }
    *task_h                    = &task->super;
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
