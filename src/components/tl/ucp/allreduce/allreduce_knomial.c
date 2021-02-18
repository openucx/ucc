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

#define RESTORE_STATE()                                                        \
    do {                                                                       \
        iteration = task->allreduce_kn.iteration;                              \
        radix_pow = task->allreduce_kn.radix_mask_pow;                         \
    } while (0)

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allreduce_kn.phase          = _phase;                            \
        task->allreduce_kn.iteration      = iteration;                         \
        task->allreduce_kn.radix_mask_pow = radix_pow;                         \
    } while (0)

ucc_status_t ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = task->team;
    int                myrank     = team->rank;
    int                group_size = team->size;
    int                radix      = task->allreduce_kn.radix;
    void              *scratch    = task->allreduce_kn.scratch;
    void              *sbuf       = task->args.src.info.buffer;
    void              *rbuf       = task->args.dst.info.buffer;
    ucc_memory_type_t  smem       = task->args.src.info.mem_type;
    ucc_memory_type_t  rmem       = task->args.dst.info.mem_type;
    size_t             count      = task->args.dst.info.count;
    ucc_datatype_t     dt         = task->args.dst.info.datatype;
    size_t             data_size  = count * ucc_dt_size(dt);
    ucc_memory_type_t  send_mem;
    void              *send_buf;
    ptrdiff_t          recv_offset;
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, k, step_size, peer;

    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    if (UCC_IS_INPLACE(task->args)) {
        sbuf = rbuf;
    }
    GOTO_PHASE(task->allreduce_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
        ucc_tl_ucp_send_nb(sbuf, data_size, smem, peer, team, task);
        ucc_tl_ucp_recv_nb(rbuf, data_size, rmem, peer, team, task);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        ucc_tl_ucp_recv_nb(scratch, data_size, smem, peer, team, task);
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
            ucc_dt_reduce(sbuf, scratch, rbuf,
                          count, dt, UCC_MEMORY_TYPE_HOST, &task->args);
        }
    }
    for (; iteration < pow_k_sup; iteration++) {
        step_size = radix_pow * radix;
        for (k = 1; k < radix; k++) {
            peer = (myrank + k * radix_pow) % step_size +
                   (myrank - myrank % step_size);
            if (peer >= full_size)
                continue;
            if ((iteration == 0) && (KN_NODE_PROXY != node_type) &&
                !UCC_IS_INPLACE(task->args)) {
                send_buf = sbuf;
                send_mem = smem;
            } else {
                send_buf = rbuf;
                send_mem = rmem;
            }
            ucc_tl_ucp_send_nb(send_buf, data_size, send_mem, peer, team, task);
        }

        recv_offset = 0;
        for (k = 1; k < radix; k++) {
            peer = (myrank + k * radix_pow) % step_size +
                   (myrank - myrank % step_size);
            if (peer >= full_size)
                continue;
            ucc_tl_ucp_recv_nb((void*)((ptrdiff_t)scratch + recv_offset),
                               data_size, smem, peer, team, task);
            recv_offset += data_size;
        }
        radix_pow *= radix;
    PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(PHASE_LOOP);
            return task->super.super.status;
        }

        if (task->send_posted > iteration * (radix - 1)) {
            if ((iteration == 0) && (KN_NODE_PROXY != node_type) &&
                !UCC_IS_INPLACE(task->args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            ucc_dt_reduce_multi(
                send_buf, scratch, rbuf,
                task->send_posted - iteration * (radix - 1), count, data_size,
                dt, UCC_MEMORY_TYPE_HOST, &task->args);
        }
    }
    if (KN_NODE_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        ucc_tl_ucp_send_nb(rbuf, data_size, rmem, peer, team, task);
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
    task->allreduce_kn.iteration      = 0;
    task->allreduce_kn.radix_mask_pow = 1;
    task->allreduce_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_CTX(team)->cfg.allreduce_kn_radix, team->size);
    ucc_mc_alloc(&task->allreduce_kn.scratch,
                 (task->allreduce_kn.radix - 1) * data_size,
                 task->args.src.info.mem_type);

    task->super.super.status = UCC_INPROGRESS;
    status = ucc_tl_ucp_allreduce_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (status < 0) {
        return status;
    }
    return UCC_OK;
}
