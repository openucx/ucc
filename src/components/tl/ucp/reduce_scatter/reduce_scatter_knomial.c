/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp_reduce.h"
#include "tl_ucp_sendrecv.h"
#include "reduce_scatter.h"
#include "core/ucc_progress_queue.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_scatter_kn.phase = _phase;                                \
    } while (0)


#define GET_COUNT(_args, _size)                                         \
    ((_args).coll_type == UCC_COLL_TYPE_ALLREDUCE ?                     \
     (_args).dst.info.count : (_args).dst.info.count * (_size))


/* Buffer for send operations at given iteration of main knomial loop.
   On the first iteration send data can be in several locations:
   - If the calling rank served another "extra" rank then the result of first
   reduction is in the "user dst" buffer.
   - INPLACE case: the data is originally in "user dst"
   After first iteration the data to send (the result of reduction from previous
   iteration) will be in the beginning of the scratch buffer.
   Note, scratch can point to "user dst" if NON_EXTRA case && NON_INPLACE.
   This is the only case when scratch allocation is not required. */
#define GET_SBUF(_task, _node_type, _args)                                    \
    ((ucc_knomial_pattern_loop_first_iteration(&(_task)->reduce_scatter_kn.p)) \
        ? ((KN_NODE_PROXY ==  (_node_type) || UCC_IS_INPLACE(*(_args)))       \
            ? (_args)->dst.info.buffer                                        \
            : (_args)->src.info.buffer)                                       \
     : (_task)->reduce_scatter_kn.scratch)

/* Buffer for recv operations at given iteration of main knomial loop.
   On the first iteration it points the beginning of the scratch.
   On all subsequent iterations we store the result of the reduction from
   previous iterations in the beginning of the scratch. It occupies
   block_count * dt_size bytes - we need to offset. */
#define GET_RBUF(_task, _block_size)                                          \
    ((!ucc_knomial_pattern_loop_first_iteration(&(_task)->reduce_scatter_kn.p)) \
        ? PTR_OFFSET((_task)->reduce_scatter_kn.scratch,                      \
                 (_block_size))                                               \
    : (_task)->reduce_scatter_kn.scratch)

static inline ucc_status_t reduce_blocks(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t       *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;
    int                    avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    ucc_knomial_pattern_t *p          = &task->reduce_scatter_kn.p;
    uint8_t                node_type  = p->node_type;
    ucc_memory_type_t      mem_type   = args->dst.info.mem_type;
    ucc_datatype_t         dt         = args->dst.info.datatype;
    size_t                 dt_size    = ucc_dt_size(dt);
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    size_t                 count      = GET_COUNT(*args, size);
    size_t                 block_count, local_seg_count, local_seg_offset;
    void                  *local_block, *recv_blocks, *reduce_target;
    int                    n_recv, is_avg;
    ucc_rank_t             step_radix;

    n_recv = task->send_posted - p->iteration * (radix - 1);
    if (n_recv > 0) {
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);

        ucc_sra_kn_peer_block_seg(p, rank, step_radix, block_count,
                                  &local_seg_count, &local_seg_offset);

        local_block  = PTR_OFFSET(GET_SBUF(task, node_type, args),
                                  local_seg_offset * dt_size);
        recv_blocks  = GET_RBUF(task, block_count * dt_size);
        reduce_target = task->reduce_scatter_kn.scratch;
        if (task->reduce_scatter_kn.scratch != args->dst.info.buffer &&
                ucc_knomial_pattern_loop_last_iteration(p)) {
            if (args->coll_type != UCC_COLL_TYPE_ALLREDUCE) {
                local_seg_offset = 0;
            } else {
                ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix,
                                                 &local_seg_offset, &local_seg_count);
            }
            reduce_target = PTR_OFFSET(args->dst.info.buffer, local_seg_offset);
        }

        is_avg = args->reduce.predefined_op == UCC_OP_AVG &&
            (avg_pre_op ? ucc_knomial_pattern_loop_first_iteration(p)
             : ucc_knomial_pattern_loop_last_iteration(p));

        return ucc_tl_ucp_reduce_multi(local_block, recv_blocks, reduce_target,
                                       n_recv, local_seg_count, local_seg_count * dt_size,
                                       dt, mem_type, task, is_avg);
    }
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;
    ucc_knomial_pattern_t *p     = &task->reduce_scatter_kn.p;
    uint8_t                node_type  = p->node_type;
    void                  *scratch    = task->reduce_scatter_kn.scratch;
    ucc_memory_type_t      mem_type   = args->dst.info.mem_type;
    ucc_datatype_t         dt         = args->dst.info.datatype;
    size_t                 dt_size    = ucc_dt_size(dt);
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    size_t                 count      = GET_COUNT(*args, size);
    void                  *rbuf       = args->dst.info.buffer;
    void                  *sbuf       = UCC_IS_INPLACE(*args)
        ? args->dst.info.buffer : args->src.info.buffer;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer, step_radix;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;

    UCC_KN_GOTO_PHASE(task->reduce_scatter_kn.phase);

    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(sbuf, count * dt_size, mem_type, peer, team, task),
            task, out);
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(scratch, count * dt_size, mem_type, peer, team, task),
            task, out);
    }

UCC_KN_PHASE_EXTRA:
    if ((KN_NODE_PROXY == node_type) || (KN_NODE_EXTRA == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return task->super.super.status;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        } else {
            if (UCC_OK != (status = ucc_dt_reduce(sbuf, scratch, rbuf, count,
                                                  dt, mem_type, args))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
    }
    while (!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        sbuf        = GET_SBUF(task, node_type, args);

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;

            ucc_sra_kn_peer_block_seg(p, peer, step_radix, block_count,
                                      &peer_seg_count, &peer_seg_offset);
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
        }

        ucc_sra_kn_peer_block_seg(p, rank, step_radix, block_count,
                                  &local_seg_count, &local_seg_offset);

        rbuf = GET_RBUF(task, block_count * dt_size);
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
            return task->super.super.status;
        }
        status = reduce_blocks(task);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
            task->super.super.status = status;
            return status;
        }
        ucc_knomial_pattern_next_iteration(p);
    }

    if (task->reduce_scatter_kn.scratch == args->dst.info.buffer) {
        ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix,
                                         &local_seg_offset, &local_seg_count);

        status = ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer, local_seg_offset),
                               task->reduce_scatter_kn.scratch,
                               local_seg_count * dt_size, mem_type, mem_type);

        if (UCC_OK != status) {
            return status;
        }
    }
UCC_KN_PHASE_PROXY: /* unused label */
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_done",
                                     0);
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_start",
                                     0);
    ucc_tl_ucp_task_reset(task);

    ucc_knomial_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                             &task->reduce_scatter_kn.p);
    if (!task->reduce_scatter_kn.scratch_mc_header) {
        task->reduce_scatter_kn.scratch = args->dst.info.buffer;
    }
    task->reduce_scatter_kn.phase = UCC_KN_PHASE_INIT;

    status = ucc_tl_ucp_reduce_scatter_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
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
    size_t             count     = GET_COUNT(coll_args->args, size);
    ucc_datatype_t     dt        = coll_args->args.dst.info.datatype;
    size_t             dt_size   = ucc_dt_size(dt);
    size_t             data_size = count * dt_size;
    ucc_memory_type_t  mem_type  = coll_args->args.dst.info.mem_type;
    ucc_knomial_pattern_t p;
    ucc_tl_ucp_task_t    *task;
    ucc_status_t          status;

    ucc_knomial_pattern_init(size, rank, radix, &p);
    if (coll_args->args.coll_type == UCC_COLL_TYPE_REDUCE_SCATTER &&
        p.n_extra > 0) {
        /* ReduceScatter Knomial currently supports only full kn tree
           case - no extra. fallback to ring. */
        return ucc_tl_ucp_reduce_scatter_ring_init(coll_args, team, task_h);
    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);
    task->reduce_scatter_kn.scratch_mc_header = NULL;
    task->reduce_scatter_kn.p                 = p;
    /* Scratch allocation can be skipped only when:
       allreduce_sra_no_scratch params is set to "Y" - we want to try avoiding
       scratch space allocation           &&
       algorithm is ALLREDUCE (coll_type) &&
       NON_INPLACE (user provided dst buffer which can be used as scratch) &&
       i'm not a proxy rank (otherwise dst buffer already contains result of
       communication/reduction with "extra") */
    if (!UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sra_kn_no_scratch ||
        coll_args->args.coll_type == UCC_COLL_TYPE_REDUCE_SCATTER      ||
        UCC_IS_INPLACE(coll_args->args)                                ||
        KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) {
        status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch_mc_header,
                              data_size, mem_type);
        task->reduce_scatter_kn.scratch =
            task->reduce_scatter_kn.scratch_mc_header->addr;
        if (UCC_OK != status) {
            return status;
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
    size_t             count   = GET_COUNT(coll_args->args, size);
    ucc_kn_radix_t     radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatter_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix, size, count);
    return ucc_tl_ucp_reduce_scatter_knomial_init_r(coll_args, team, task_h,
                                                    radix);
}
