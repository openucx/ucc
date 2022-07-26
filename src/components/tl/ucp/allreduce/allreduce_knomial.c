/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/recursive_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/ec/ucc_ec.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->allreduce_kn.phase = _phase;                                     \
    } while (0)

void ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    int                    avg_pre_op = UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    ucc_kn_radix_t         radix      = task->allreduce_kn.p.radix;
    uint8_t                node_type  = task->allreduce_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->allreduce_kn.p;
    void                  *scratch    = task->allreduce_kn.scratch;
    void                  *sbuf       = args->src.info.buffer;
    void                  *rbuf       = args->dst.info.buffer;
    ucc_memory_type_t      mem_type   = args->dst.info.mem_type;
    size_t                 count      = args->dst.info.count;
    ucc_datatype_t         dt         = args->dst.info.datatype;
    size_t                 data_size  = count * ucc_dt_size(dt);
    ucc_rank_t             size       = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t             rank       = task->subset.myrank;
    void                  *send_buf;
    ptrdiff_t              recv_offset;
    ucc_rank_t             peer;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    int                    is_avg;
    ucc_ee_executor_t     *exec;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        task->super.status = status;
        return;
    }
    if (UCC_IS_INPLACE(*args)) {
        sbuf = rbuf;
    }
    UCC_KN_REDUCE_GOTO_PHASE(task->allreduce_kn.phase);

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
            return;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto completion;
        } else {
            status = ucc_dt_reduce_nb(sbuf, scratch, rbuf, count, dt, args,
                                      exec, &task->allreduce_kn.etask);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
UCC_KN_PHASE_EXTRA_REDUCE:
            EXEC_TASK_TEST(UCC_KN_PHASE_EXTRA_REDUCE,
                           "failed to perform dt reduction",
                           task->allreduce_kn.etask);
        }
    }
    while(!ucc_knomial_pattern_loop_done(p)) {
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            peer = ucc_ep_map_eval(task->subset.map, peer);
            if ((ucc_knomial_pattern_loop_first_iteration(p)) &&
                (KN_NODE_PROXY != node_type) && !UCC_IS_INPLACE(*args)) {
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
            return;
        }

        if (task->tagged.send_posted > p->iteration * (radix - 1)) {
            if ((ucc_knomial_pattern_loop_first_iteration(p)) &&
                (KN_NODE_PROXY != node_type) && !UCC_IS_INPLACE(*args)) {
                send_buf = sbuf;
            } else {
                send_buf = rbuf;
            }
            is_avg = args->op == UCC_OP_AVG &&
                     (avg_pre_op ? ucc_knomial_pattern_loop_first_iteration(p)
                                 : ucc_knomial_pattern_loop_last_iteration(p));
            status = ucc_tl_ucp_reduce_multi_nb(
                send_buf, scratch, rbuf,
                task->tagged.send_posted - p->iteration * (radix - 1), count,
                data_size, dt, task, is_avg, exec,
                &task->allreduce_kn.etask);
            if (ucc_unlikely(UCC_OK != status)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
UCC_KN_PHASE_REDUCE:
            EXEC_TASK_TEST(UCC_KN_PHASE_REDUCE,
                           "failed to perform dt reduction",
                           task->allreduce_kn.etask);
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
        return;
    }

completion:
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_kn_done", 0);
UCC_KN_PHASE_COMPLETE: /* unused label */
out:
    return;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         size      = (ucc_rank_t)task->subset.map.ep_num;
    ucc_rank_t         rank      = task->subset.myrank;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_kn_start", 0);
    task->allreduce_kn.phase = UCC_KN_PHASE_INIT;
    ucc_assert(UCC_IS_INPLACE(TASK_ARGS(task)) ||
               (TASK_ARGS(task).src.info.mem_type ==
               TASK_ARGS(task).dst.info.mem_type));
    ucc_knomial_pattern_init(size, rank,
                             ucc_min(UCC_TL_UCP_TEAM_LIB(team)->
                                     cfg.allreduce_kn_radix, size),
                             &task->allreduce_kn.p);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_init_common(ucc_tl_ucp_task_t *task)
{
    size_t             count     = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_rank_t         size      = (ucc_rank_t)task->subset.map.ep_num;
    ucc_kn_radix_t     radix =
        ucc_min(TASK_LIB(task)->cfg.allreduce_kn_radix, size);
    ucc_status_t       status;

    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_allreduce_knomial_start;
    task->super.progress = ucc_tl_ucp_allreduce_knomial_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_knomial_finalize;
    status               = ucc_mc_alloc(&task->allreduce_kn.scratch_mc_header,
                          (radix - 1) * data_size,
                          TASK_ARGS(task).dst.info.mem_type);
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
