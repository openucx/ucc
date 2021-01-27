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
enum {
    PHASE_0,
    PHASE_1,
    PHASE_EXTRA,
    PHASE_PROXY,
};

#define CHECK_PHASE(_p) case _p: goto _p; break;
#define GOTO_PHASE(_phase) do{                  \
        switch (_phase) {                       \
            CHECK_PHASE(PHASE_EXTRA);           \
            CHECK_PHASE(PHASE_PROXY);           \
            CHECK_PHASE(PHASE_1);               \
        case PHASE_0: break;                    \
        };                                      \
    } while(0)

#define RESTORE_STATE() do{                             \
        iteration   = task->barrier.iteration;         \
        radix_pow   = task->barrier.radix_mask_pow;    \
    }while(0)

#define SAVE_STATE(_phase) do{                         \
        task->barrier.phase          = _phase;        \
        task->barrier.iteration      = iteration;     \
        task->barrier.radix_mask_pow = radix_pow;     \
    }while(0)

ucc_status_t ucc_tl_ucp_barrier_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    int myrank              = team->rank;
    int group_size          = team->size;
    int radix               = task->barrier.radix;
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, k, step_size, peer;

    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    GOTO_PHASE(task->barrier.phase);

    if (KN_NODE_EXTRA == node_type) {
            peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
            ucc_tl_ucp_send_nb(NULL, 0, UCS_MEMORY_TYPE_UNKNOWN, peer,
                               team, task->tag, task);
            ucc_tl_ucp_recv_nb(NULL, 0, UCS_MEMORY_TYPE_UNKNOWN, peer,
                               team, task->tag, task);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        ucc_tl_ucp_recv_nb(NULL, 0, UCS_MEMORY_TYPE_UNKNOWN, peer,
                           team, task->tag, task);
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

    for (; iteration < pow_k_sup; iteration++) {
        step_size = radix_pow * radix;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            ucc_tl_ucp_send_nb(NULL, 0, UCS_MEMORY_TYPE_UNKNOWN, peer,
                               team, task->tag, task);
        }

        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            ucc_tl_ucp_recv_nb(NULL, 0, UCS_MEMORY_TYPE_UNKNOWN, peer,
                               team, task->tag, task);
        }
        radix_pow *= radix;
    PHASE_1:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_1);
            return UCC_INPROGRESS;
        }
    }
    if (KN_NODE_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        ucc_tl_ucp_send_nb(NULL, 0, UCS_MEMORY_TYPE_UNKNOWN, peer,
                           team, task->tag, task);
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
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    ucc_status_t       status;
    task->barrier.phase          = PHASE_0;
    task->barrier.iteration      = 0;
    task->barrier.radix_mask_pow = 1;
    task->barrier.radix          = 4; //TODO env var
    if (task->barrier.radix > team->size) {
        task->barrier.radix = team->size;
    }

    status = ucc_tl_ucp_barrier_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (status < 0) {
        return status;
    }
    return UCC_OK;
}
