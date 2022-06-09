/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "reduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_dt_reduce.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_kn.phase = _phase;                                        \
    } while (0)

void ucc_tl_ucp_reduce_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task,
                                                   ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    int                avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size       = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         root       = (ucc_rank_t)args->root;
    uint32_t           radix      = task->reduce_kn.radix;
    ucc_rank_t         vrank      = (rank - root + size) % size;
    void              *rbuf       = (rank == root) ? args->dst.info.buffer :
                                                      task->reduce_kn.scratch;
    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    void              *received_vectors, *scratch_offset;
    ucc_rank_t         vpeer, peer, vroot_at_level, root_at_level, pos;
    uint32_t           i;
    ucc_status_t       status;
    int                is_avg;

    if (root == rank) {
        count = args->dst.info.count;
        data_size = count * ucc_dt_size(args->dst.info.datatype);
        mtype = args->dst.info.mem_type;
        dt = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        data_size = count * ucc_dt_size(args->src.info.datatype);
        mtype = args->src.info.mem_type;
        dt = args->src.info.datatype;
    }
    received_vectors = PTR_OFFSET(task->reduce_kn.scratch, data_size);

UCC_REDUCE_KN_PHASE_PROGRESS:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    UCC_REDUCE_KN_GOTO_PHASE(task->reduce_kn.phase);

UCC_REDUCE_KN_PHASE_INIT:

    while (task->reduce_kn.dist <= task->reduce_kn.max_dist) {
        if (vrank % task->reduce_kn.dist == 0) {
            pos = (vrank / task->reduce_kn.dist) % radix;
            if (pos == 0) {
                scratch_offset = received_vectors;
                task->reduce_kn.children_per_cycle = 0;
                for (i = 1; i < radix; i++) {
                    vpeer = vrank + i * task->reduce_kn.dist;
                    if (vpeer >= size) {
                    	break;
                    } else {
                        task->reduce_kn.children_per_cycle += 1;
                        peer = (vpeer + root) % size;
                        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                          data_size, mtype, peer, team, task),
                                          task, out);
                        scratch_offset = PTR_OFFSET(scratch_offset, data_size);
                    }
                }
                SAVE_STATE(UCC_REDUCE_KN_PHASE_MULTI);
                goto UCC_REDUCE_KN_PHASE_PROGRESS;
UCC_REDUCE_KN_PHASE_MULTI:
                if (task->reduce_kn.children_per_cycle) {
                    is_avg = args->op == UCC_OP_AVG &&
                             (avg_pre_op ? (task->reduce_kn.dist == 1)
                                         : (task->reduce_kn.dist ==
                                            task->reduce_kn.max_dist));
                    status = ucc_dt_reduce_strided(
                        (task->reduce_kn.dist == 1) ? args->src.info.buffer
                                                    : rbuf,
                        received_vectors, rbuf,
                        task->reduce_kn.children_per_cycle, count, data_size,
                        dt, args,
                        is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                        AVG_ALPHA(task), task->reduce_kn.executor,
                        &task->reduce_kn.etask);
                    if (ucc_unlikely(UCC_OK != status)) {
                        tl_error(UCC_TASK_LIB(task),
                                 "failed to perform dt reduction");
                        task->super.status = status;
                        return;
                    }
                    EXEC_TASK_WAIT(task->reduce_kn.etask);
                }
            } else {
                vroot_at_level = vrank - pos * task->reduce_kn.dist;
                root_at_level  = (vroot_at_level + root) % size;
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->reduce_kn.scratch,
                                  data_size, mtype, root_at_level, team, task),
                                  task, out);
            }
        }
        task->reduce_kn.dist *= radix;
        SAVE_STATE(UCC_REDUCE_KN_PHASE_INIT);
        goto UCC_REDUCE_KN_PHASE_PROGRESS;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_kn_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_reduce_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    uint32_t           radix      = task->reduce_kn.radix;
    ucc_rank_t         root       = (ucc_rank_t)args->root;
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size       = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         vrank      = (rank - root + size) % size;
    int                isleaf     =
        (vrank % radix != 0 || vrank == size - 1);
    int                avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    int                self_avg   = (args->op == UCC_OP_AVG &&
        avg_pre_op && vrank % radix == 0);
    size_t             count;
    ucc_datatype_t     dt;
    ucc_status_t       status;

    if (root == rank) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
    }

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (UCC_IS_INPLACE(*args) && (rank == root)) {
        args->src.info.buffer = args->dst.info.buffer;
    }

    if (isleaf && !self_avg) {
    	task->reduce_kn.scratch = args->src.info.buffer;
    }

    status =
        ucc_coll_task_get_executor(&task->super, &task->reduce_kn.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    if (isleaf && self_avg) {
        /* In case of avg_pre_op, single leaf process which does not take part
           in first iteration reduction must divide itself by team_size */

        status =
            ucc_dt_reduce(args->src.info.buffer, args->src.info.buffer,
                          task->reduce_kn.scratch, count, dt, args,
                          UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA,
                          1.0 / (double)(UCC_TL_TEAM_SIZE(TASK_TEAM(task)) * 2),
                          task->reduce_kn.executor, &task->reduce_kn.etask);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to perform dt reduction");
            task->super.status = status;
            return status;
        }
        EXEC_TASK_WAIT(task->reduce_kn.etask, status);
    }

    task->reduce_kn.dist = 1;
    task->reduce_kn.phase = UCC_REDUCE_KN_PHASE_INIT;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_reduce_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->reduce_kn.scratch_mc_header) {
        ucc_mc_free(task->reduce_kn.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}
