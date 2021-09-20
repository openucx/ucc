/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/recursive_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "core/ucc_mc.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allreduce_kn.phase = _phase;                                     \
    } while (0)

ucc_status_t ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args = &coll_task->args;
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_kn_radix_t         radix     = task->allreduce_kn.p.radix;
    uint8_t                node_type = task->allreduce_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->allreduce_kn.p;
    void                  *scratch   = task->allreduce_kn.scratch;
    void                  *sbuf      = args->src.info.buffer;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->dst.info.mem_type;
    size_t                 count     = args->dst.info.count;
    ucc_datatype_t         dt        = args->dst.info.datatype;
    size_t                 data_size = count * ucc_dt_size(dt);
    ucc_rank_t             size      = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t             rank      = task->subset.myrank;
    void                  *send_buf;
    ptrdiff_t              recv_offset;
    ucc_rank_t             peer;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    if (UCC_IS_INPLACE(*args)) {
        sbuf = rbuf;
    }
    UCC_KN_GOTO_PHASE(task->allreduce_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_ep_map_eval(task->subset.map,
                               ucc_knomial_pattern_get_proxy(p, rank));
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(sbuf, data_size, mem_type, peer, team, task),
            task, out);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(rbuf, data_size, mem_type, peer, team, task),
            task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_ep_map_eval(task->subset.map,
                               ucc_knomial_pattern_get_extra(p, rank));
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(scratch, data_size, mem_type, peer, team, task),
            task, out);
    }
UCC_KN_PHASE_EXTRA:
    if (KN_NODE_PROXY == node_type || KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return task->super.super.status;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto completion;
        } else {
            if (ucc_unlikely(UCC_OK !=
                             (status = ucc_dt_reduce(sbuf, scratch, rbuf, count,
                                                     dt, mem_type, args)))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
    }
    while(!ucc_knomial_pattern_loop_done(p)) {
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            peer = ucc_ep_map_eval(task->subset.map, peer);
            if ((p->iteration == 0) && (KN_NODE_PROXY != node_type) &&
                !UCC_IS_INPLACE(*args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(send_buf, data_size, mem_type, peer, team,
                                   task),
                task, out);
        }

        recv_offset = 0;
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            peer = ucc_ep_map_eval(task->subset.map, peer);
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb((void *)((ptrdiff_t)scratch + recv_offset),
                                   data_size, mem_type, peer, team, task),
                task, out);
            recv_offset += data_size;
        }

    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return task->super.super.status;
        }

        if (task->send_posted > p->iteration * (radix - 1)) {
            if ((p->iteration == 0) && (KN_NODE_PROXY != node_type) &&
                !UCC_IS_INPLACE(*args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            if (ucc_unlikely(
                    UCC_OK !=
                    (status = ucc_dt_reduce_multi(
                         send_buf, scratch, rbuf,
                         task->send_posted - p->iteration * (radix - 1), count,
                         data_size, dt, mem_type, args)))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
        ucc_knomial_pattern_next_iteration(p);
    }
    if (KN_NODE_PROXY == node_type) {
        peer = ucc_ep_map_eval(task->subset.map,
                               ucc_knomial_pattern_get_extra(p, rank));
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(rbuf, data_size, mem_type, peer, team, task),
            task, out);
        goto UCC_KN_PHASE_PROXY;
    } else {
        goto completion;
    }

UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return task->super.super.status;
    }

completion:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.super.status = UCC_OK;
out:
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         size      = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         rank      = task->subset.myrank;
    ucc_status_t       status;

    task->allreduce_kn.phase = UCC_KN_PHASE_INIT;
    ucc_assert(coll_task->args.src.info.mem_type ==
               coll_task->args.dst.info.mem_type);
    ucc_knomial_pattern_init(size, rank,
                             ucc_min(UCC_TL_UCP_TEAM_LIB(team)->
                                     cfg.allreduce_kn_radix, size),
                             &task->allreduce_kn.p);
    ucc_tl_ucp_task_reset(task);
    task->super.super.status = UCC_INPROGRESS;
    status = ucc_tl_ucp_allreduce_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_init_common(ucc_tl_ucp_task_t *task)
{
    size_t             count     = task->super.args.dst.info.count;
    ucc_datatype_t     dt        = task->super.args.dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_rank_t         size      = (ucc_rank_t)task->subset.map.ep_num;
    ucc_kn_radix_t     radix =
        ucc_min(TASK_LIB(task)->cfg.allreduce_kn_radix, size);
    ucc_status_t       status;

    task->super.post     = ucc_tl_ucp_allreduce_knomial_start;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_knomial_finalize;
    status               = ucc_mc_alloc(&task->allreduce_kn.scratch_mc_header,
                          (radix - 1) * data_size,
                          task->super.args.dst.info.mem_type);
    task->allreduce_kn.scratch = task->allreduce_kn.scratch_mc_header->addr;
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
        return status;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t st, global_st;

    global_st = ucc_mc_free(task->allreduce_kn.scratch_mc_header);
    if (ucc_unlikely(global_st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to free scratch buffer");
    }

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
        global_st = st;
    }
    return global_st;
}
