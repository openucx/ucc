/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "barrier.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/recursive_knomial.h"
#include "utils/ucc_math.h"
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
        task->barrier.phase = _phase;                                 \
    } while (0)

ucc_status_t ucc_tl_ucp_barrier_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team       = task->team;
    ucc_kn_radix_t         radix      = task->barrier.p.radix;
    uint8_t                node_type  = task->barrier.p.node_type;
    ucc_knomial_pattern_t *p          = &task->barrier.p;
    ucc_memory_type_t      mtype      = UCC_MEMORY_TYPE_UNKNOWN;
    ucc_rank_t             peer;
    ucc_kn_radix_t         loop_step;

    GOTO_PHASE(task->barrier.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, team->rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, team->rank);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
    }
PHASE_EXTRA:
    if (KN_NODE_PROXY == node_type || KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_EXTRA);
            return UCC_INPROGRESS;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto completion;
        }
    }

    while(!ucc_knomial_pattern_loop_done(p)) {
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, team->rank,
                                                     team->size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(NULL, 0, mtype, peer, team, task),
                          task, out);
        }

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, team->rank,
                                                     team->size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(NULL, 0, mtype, peer, team, task),
                          task, out);
        }
    PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_LOOP);
            return UCC_INPROGRESS;
        }
        ucc_knomial_pattern_next_iteration(p);
    }
    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, team->rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
        goto PHASE_PROXY;
    } else {
        goto completion;
    }

PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(PHASE_PROXY);
        return UCC_INPROGRESS;
    }

completion:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.super.status = UCC_OK;
out:
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    ucc_status_t       status;
    task->barrier.phase          = PHASE_INIT;
    ucc_knomial_pattern_init(team->size, team->rank,
                             ucc_min(UCC_TL_UCP_TEAM_LIB(team)->
                                     cfg.barrier_kn_radix, team->size),
                             &task->barrier.p);
    task->super.super.status = UCC_INPROGRESS;
    status = ucc_tl_ucp_barrier_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}
