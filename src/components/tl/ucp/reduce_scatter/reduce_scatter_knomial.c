/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp_reduce.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_scatter_kn.phase = _phase;                                \
    } while (0)

void ucc_tl_ucp_reduce_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;
    int                    avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    uint8_t                node_type  = task->reduce_scatter_kn.p.node_type;
    ucc_knomial_pattern_t *p          = &task->reduce_scatter_kn.p;
    void                  *scratch    = task->reduce_scatter_kn.scratch;
    void                  *rbuf       = args->dst.info.buffer;
    ucc_memory_type_t      mem_type   = args->dst.info.mem_type;
    size_t                 count      = args->dst.info.count;
    ucc_datatype_t         dt         = args->dst.info.datatype;
    void                  *sbuf       = UCC_IS_INPLACE(*args) ?
        rbuf : args->src.info.buffer;
    size_t                 dt_size    = ucc_dt_size(dt);
    size_t                 data_size  = count * dt_size;
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    ptrdiff_t              peer_seg_offset, local_seg_offset, offset;
    ucc_rank_t             peer, step_radix, peer_seg_index, local_seg_index;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;
    void                  *reduce_data, *local_data;
    int                    is_avg;
    ucc_ee_executor_t     *exec;
    ucc_ee_executor_task_args_t eargs;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        task->super.status = status;
        return;
    }
    local_seg_count = 0;
    block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
    UCC_KN_REDUCE_GOTO_PHASE(task->reduce_scatter_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(sbuf, data_size, mem_type, peer, team, task),
            task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(scratch, data_size, mem_type, peer, team, task),
            task, out);
    }

UCC_KN_PHASE_EXTRA:
    if ((KN_NODE_PROXY == node_type) || (KN_NODE_EXTRA == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        } else {
            status = ucc_dt_reduce_nb(sbuf, scratch, rbuf, count, dt, args,
                                      exec, &task->reduce_scatter_kn.etask);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
UCC_KN_PHASE_EXTRA_REDUCE:
            EXEC_TASK_TEST(UCC_KN_PHASE_EXTRA_REDUCE,
                           "failed to perform dt reduction",
                           task->reduce_scatter_kn.etask);
        }
    }
    while (!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        sbuf        = (ucc_knomial_pattern_loop_first_iteration(p))
                          ? ((KN_NODE_PROXY == node_type || UCC_IS_INPLACE(*args))
                                 ? args->dst.info.buffer
                                 : args->src.info.buffer)
                          : task->reduce_scatter_kn.scratch;
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            peer_seg_index =
                ucc_sra_kn_compute_seg_index(peer, p->radix_pow, p);
            peer_seg_count = ucc_sra_kn_compute_seg_size(
                block_count, step_radix, peer_seg_index);
            peer_seg_offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, peer_seg_index);
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
        }

        local_seg_index = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
        local_seg_count = ucc_sra_kn_compute_seg_size(block_count, step_radix,
                                                      local_seg_index);

        rbuf = task->reduce_scatter_kn.scratch;
        if (!ucc_knomial_pattern_loop_first_iteration(p)) {
            rbuf = PTR_OFFSET(rbuf, block_count * dt_size);
        }
        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(rbuf, local_seg_count * dt_size,
                                             mem_type, peer, team, task),
                          task, out);
            rbuf = PTR_OFFSET(rbuf, local_seg_count * dt_size);
        }
    UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        if (task->tagged.send_posted > p->iteration * (radix - 1)) {
            sbuf       = (ucc_knomial_pattern_loop_first_iteration(p))
                             ? ((KN_NODE_PROXY == node_type || UCC_IS_INPLACE(*args))
                                    ? args->dst.info.buffer
                                    : args->src.info.buffer)
                             : task->reduce_scatter_kn.scratch;
            rbuf       = (!ucc_knomial_pattern_loop_first_iteration(p))
                             ? PTR_OFFSET(task->reduce_scatter_kn.scratch,
                                    block_count * dt_size)
                             : task->reduce_scatter_kn.scratch;
            step_radix = ucc_sra_kn_compute_step_radix(rank, size, p);
            local_seg_index =
                ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
            local_seg_count = ucc_sra_kn_compute_seg_size(
                block_count, step_radix, local_seg_index);
            local_seg_offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, local_seg_index);
            local_data  = PTR_OFFSET(sbuf, local_seg_offset * dt_size);
            reduce_data = task->reduce_scatter_kn.scratch;
            is_avg      = args->op == UCC_OP_AVG &&
                     (avg_pre_op ? ucc_knomial_pattern_loop_first_iteration(p)
                                 : ucc_knomial_pattern_loop_last_iteration(p));

            status = ucc_tl_ucp_reduce_multi_nb(
                local_data, rbuf, reduce_data,
                task->tagged.send_posted - p->iteration * (radix - 1),
                local_seg_count, local_seg_count * dt_size, dt, task,
                is_avg, exec, &task->reduce_scatter_kn.etask);
            if (ucc_unlikely(UCC_OK != status)) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.status = status;
                return;
            }
UCC_KN_PHASE_REDUCE:
            EXEC_TASK_TEST(UCC_KN_PHASE_REDUCE,
                           "failed to perform dt reduction",
                           task->reduce_scatter_kn.etask);
        }
        ucc_knomial_pattern_next_iteration(p);
    }

    ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix,
                                     &offset, &local_seg_count);
    eargs.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
    eargs.bufs[0]   = PTR_OFFSET(args->dst.info.buffer, offset);
    eargs.bufs[1]   = task->reduce_scatter_kn.scratch;
    eargs.count     = local_seg_count * dt_size;
    status = ucc_ee_executor_task_post(exec, &eargs,
                                       &task->reduce_scatter_kn.etask);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to copy data to dst buffer");
        task->super.status = status;
        return;
    }
UCC_KN_PHASE_COMPLETE:
    EXEC_TASK_TEST(UCC_KN_PHASE_COMPLETE, "failed to perform memcpy",
                   task->reduce_scatter_kn.etask);

UCC_KN_PHASE_PROXY: /* unused label */
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_done",
                                     0);
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    ucc_knomial_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                             &task->reduce_scatter_kn.p);
    if (!task->reduce_scatter_kn.scratch_mc_header) {
        task->reduce_scatter_kn.scratch = args->dst.info.buffer;
    }
    task->reduce_scatter_kn.phase = UCC_KN_PHASE_INIT;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->reduce_scatter_kn.scratch_mc_header) {
        ucc_mc_free(task->reduce_scatter_kn.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         rank      = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t         size      = UCC_TL_TEAM_SIZE(tl_team);
    size_t             count     = coll_args->args.dst.info.count;
    ucc_datatype_t     dt        = coll_args->args.dst.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    ucc_memory_type_t  mem_type  = coll_args->args.dst.info.mem_type;
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    size_t             max_recv_size;
    ucc_kn_radix_t     step_radix;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);
    ucc_knomial_pattern_init(size, rank, radix, &task->reduce_scatter_kn.p);
    task->reduce_scatter_kn.scratch_mc_header = NULL;

    if (KN_NODE_EXTRA != task->reduce_scatter_kn.p.node_type) {
        step_radix =
            ucc_sra_kn_compute_step_radix(rank, size,
                                          &task->reduce_scatter_kn.p);
        max_recv_size = ucc_sra_kn_compute_seg_size(count, step_radix, 0) *
            step_radix * dt_size;

        if (UCC_IS_INPLACE(coll_args->args) ||
            (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) ||
            max_recv_size > data_size) {
            status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch_mc_header,
                                  ucc_max(max_recv_size, data_size), mem_type);
            task->reduce_scatter_kn.scratch =
                task->reduce_scatter_kn.scratch_mc_header->addr;
            if (UCC_OK != status) {
                return status;
            }
        }
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(tl_team);
    size_t             count   = coll_args->args.dst.info.count;
    ucc_kn_radix_t     radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatter_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix, size, count);
    return ucc_tl_ucp_reduce_scatter_knomial_init_r(coll_args, team, task_h,
                                                    radix);
}
