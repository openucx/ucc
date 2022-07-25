/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define SAVE_STATE(_phase)                                            \
    do {                                                              \
        task->barrier.phase = _phase;                                 \
    } while (0)

void ucc_tl_ucp_barrier_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team       = TASK_TEAM(task);
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    ucc_kn_radix_t         radix      = task->barrier.p.radix;
    uint8_t                node_type  = task->barrier.p.node_type;
    ucc_knomial_pattern_t *p          = &task->barrier.p;
    ucc_memory_type_t      mtype      = UCC_MEMORY_TYPE_UNKNOWN;
    ucc_rank_t             peer;
    ucc_kn_radix_t         loop_step;


    UCC_KN_GOTO_PHASE(task->barrier.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
    }
UCC_KN_PHASE_EXTRA:
    if (KN_NODE_PROXY == node_type || KN_NODE_EXTRA == node_type) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto completion;
        }
    }

    while(!ucc_knomial_pattern_loop_done(p)) {
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank,
                                                     size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(NULL, 0, mtype, peer, team, task),
                          task, out);
        }

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank,
                                                     size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(NULL, 0, mtype, peer, team, task),
                          task, out);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        ucc_knomial_pattern_next_iteration(p);
    }
    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(NULL, 0, mtype, peer, team, task),
                      task, out);
        goto UCC_KN_PHASE_PROXY;
    } else {
        goto completion;
    }

UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return;
    }

completion:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_barrier_kn_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_barrier_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->barrier.phase = UCC_KN_PHASE_INIT;
    ucc_knomial_pattern_init(size, rank,
                             ucc_min(UCC_TL_UCP_TEAM_LIB(team)->
                                     cfg.barrier_kn_radix, size),
                             &task->barrier.p);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
