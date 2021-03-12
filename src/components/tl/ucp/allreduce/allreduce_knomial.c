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
enum {
    PHASE_INIT,
    PHASE_LOOP,  /* main loop of recursive k-ing */
    PHASE_EXTRA, /* recv from extra rank */
    PHASE_PROXY, /* send from proxy to extra rank */
};

#define CHECK_PHASE(_p)                                                        \
    case _p:                                                                   \
        goto _p;                                                               \
        break;

#define GOTO_PHASE(_phase)                                                     \
    do {                                                                       \
        switch (_phase) {                                                      \
            CHECK_PHASE(PHASE_EXTRA);                                          \
            CHECK_PHASE(PHASE_PROXY);                                          \
            CHECK_PHASE(PHASE_LOOP);                                           \
        case PHASE_INIT:                                                       \
            break;                                                             \
        };                                                                     \
    } while (0)


#define SAVE_STATE(_phase)                                            \
    do {                                                              \
        task->allreduce_kn.phase = _phase;                            \
    } while (0)

ucc_status_t ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team       = task->team;
    ucc_kn_radix_t         radix      = task->allreduce_kn.p.radix;
    uint8_t                node_type  = task->allreduce_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->allreduce_kn.p;
    void                  *scratch    = task->allreduce_kn.scratch;
    void                  *sbuf       = task->args.src.info.buffer;
    void                  *rbuf       = task->args.dst.info.buffer;
    ucc_memory_type_t      mem_type   = task->args.src.info.mem_type;
    size_t                 count      = task->args.src.info.count;
    ucc_datatype_t         dt         = task->args.src.info.datatype;
    size_t                 data_size  = count * ucc_dt_size(dt);
    void                  *send_buf;
    ptrdiff_t              recv_offset;
    ucc_rank_t             peer;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    if (UCC_IS_INPLACE(task->args)) {
        sbuf = rbuf;
    }
    GOTO_PHASE(task->allreduce_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, team->rank);
        ucc_tl_ucp_send_nb(sbuf, data_size, mem_type, peer, team, task);
        ucc_tl_ucp_recv_nb(rbuf, data_size, mem_type, peer, team, task);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, team->rank);
        ucc_tl_ucp_recv_nb(scratch, data_size, mem_type, peer, team, task);
    }
PHASE_EXTRA:
    if (KN_NODE_PROXY == node_type || KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_EXTRA);
            return task->super.super.status;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto completion;
        } else {
            if (UCC_OK != (status = ucc_dt_reduce(sbuf, scratch, rbuf,
                                                  count, dt, mem_type, &task->args))) {
                tl_error(UCC_TL_TEAM_LIB(task->team),
                         "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
    }
    while(!ucc_knomial_pattern_loop_done(p)) {
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, team->rank,
                                                     team->size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            if ((p->iteration == 0) && (KN_NODE_PROXY != node_type) &&
                !UCC_IS_INPLACE(task->args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            ucc_tl_ucp_send_nb(send_buf, data_size, mem_type, peer, team, task);
        }

        recv_offset = 0;
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, team->rank,
                                                     team->size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            ucc_tl_ucp_recv_nb((void*)((ptrdiff_t)scratch + recv_offset),
                               data_size, mem_type, peer, team, task);
            recv_offset += data_size;
        }

    PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_LOOP);
            return task->super.super.status;
        }

        if (task->send_posted > p->iteration * (radix - 1)) {
            if ((p->iteration == 0) && (KN_NODE_PROXY != node_type) &&
                !UCC_IS_INPLACE(task->args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            if (UCC_OK != (status = ucc_dt_reduce_multi(
                send_buf, scratch, rbuf,
                task->send_posted - p->iteration * (radix - 1), count, data_size,
                dt, mem_type, &task->args))) {
                tl_error(UCC_TL_TEAM_LIB(task->team),
                         "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
        ucc_knomial_pattern_next_iteration(p);
    }
    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, team->rank);
        ucc_tl_ucp_send_nb(rbuf, data_size, mem_type, peer, team, task);
        goto PHASE_PROXY;
    } else {
        goto completion;
    }

PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(PHASE_PROXY);
        return task->super.super.status;
    }

completion:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    ucc_mc_free(task->allreduce_kn.scratch, task->args.src.info.mem_type);
    task->super.super.status = UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = task->team;
    size_t             count     = task->args.src.info.count;
    ucc_datatype_t     dt        = task->args.src.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_status_t       status;
    task->allreduce_kn.phase          = PHASE_INIT;
    ucc_assert(task->args.src.info.mem_type ==
               task->args.dst.info.mem_type);
    ucc_knomial_pattern_init(team->size, team->rank,
                             ucc_min(UCC_TL_UCP_TEAM_LIB(team)->
                                     cfg.allreduce_kn_radix, team->size),
                             &task->allreduce_kn.p);
    if (UCC_OK != (status = ucc_mc_alloc(&task->allreduce_kn.scratch,
                                         (task->allreduce_kn.p.radix - 1) * data_size,
                                         task->args.src.info.mem_type))) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "failed to allocate scratch buffer");
        return status;
    }

    task->super.super.status = UCC_INPROGRESS;
    status = ucc_tl_ucp_allreduce_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (status < 0) {
        return status;
    }
    return UCC_OK;
}
