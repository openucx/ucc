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


#define GET_COUNT(_args, _size)   ({                                    \
            size_t _count = 0;                                              \
            switch ((_args)->coll_type) {                                \
            case UCC_COLL_TYPE_ALLREDUCE:                               \
                _count = (_args)->dst.info.count;                        \
                break;                                                  \
            case UCC_COLL_TYPE_REDUCE_SCATTER:                          \
                _count = UCC_IS_INPLACE(*(_args)) ? (_args)->dst.info.count : \
                (_args)->dst.info.count * (_size);                       \
                break;                                                  \
            case UCC_COLL_TYPE_REDUCE_SCATTERV:                         \
                _count = ucc_coll_args_get_total_count((_args), (_args)->dst.info_v.counts, \
                                                       (_size));        \
            default:                                                    \
                break;                                                  \
            }                                                           \
            _count;                                                     \
        })

#define GET_DATA_BUF(_args)   ({                                 \
            void *_buf = NULL;                                          \
    switch ((_args)->coll_type) {                                       \
    case UCC_COLL_TYPE_ALLREDUCE:                                       \
    case UCC_COLL_TYPE_REDUCE_SCATTER:                                  \
        _buf = UCC_IS_INPLACE(*(_args)) ? (_args)->dst.info.buffer : (_args)->src.info.buffer; \
        break;                                                          \
    case UCC_COLL_TYPE_REDUCE_SCATTERV:                                 \
        _buf = UCC_IS_INPLACE(*(_args)) ? (_args)->dst.info_v.buffer : (_args)->src.info_v.buffer; \
    default:                                                            \
        break;                                                          \
    }                                                                   \
    _buf;                                                               \
        })

#define GET_DT(_args) ((_args)->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) ? \
    (_args)->dst.info_v.datatype : (_args)->dst.info.datatype

#define GET_MT(_args) ((_args)->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) ? \
    (_args)->dst.info_v.mem_type : (_args)->dst.info.mem_type

static inline ucc_status_t reduce_blocks(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t       *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team  = TASK_TEAM(task);
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;
    int                    avg_pre_op =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op;
    ucc_knomial_pattern_t *p          = &task->reduce_scatter_kn.p;
    ucc_memory_type_t      mem_type   = GET_MT(args);
    ucc_datatype_t         dt         = GET_DT(args);
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    size_t                 dt_size    = ucc_dt_size(dt);
    size_t                 count      = GET_COUNT(args, size);
    size_t                 local_seg_count, local_seg_offset;
    void                  *local_block, *recv_blocks, *reduce_target;
    int                    n_recv, is_avg;

    n_recv = ucc_kn_compute_step_radix(rank, size, p) - 1;
    if (n_recv > 0) {
        ucc_kn_rs_pattern_peer_seg(rank, p, &local_seg_count, &local_seg_offset);
        local_block  = PTR_OFFSET(task->reduce_scatter_kn.loop_sbuf[!ucc_knomial_pattern_loop_first_iteration(p)],
                                  local_seg_offset * dt_size);
        recv_blocks  = task->reduce_scatter_kn.loop_rbuf[!ucc_knomial_pattern_loop_first_iteration(p)];
        reduce_target = task->reduce_scatter_kn.loop_sbuf[1];

        if (ucc_knomial_pattern_loop_last_iteration(p)) {
            if (args->coll_type == UCC_COLL_TYPE_ALLREDUCE &&
                task->reduce_scatter_kn.scratch_mc_header) {
                ucc_kn_rsx_pattern_dst(size, rank, radix, count,
                                       &local_seg_offset, &local_seg_count);
                reduce_target = PTR_OFFSET(args->dst.info.buffer, local_seg_offset * dt_size);
            }
            if (args->coll_type != UCC_COLL_TYPE_ALLREDUCE &&
                p->node_type != KN_NODE_PROXY) {
                reduce_target = task->reduce_scatter_kn.dest;
            }
        }
        is_avg = args->op == UCC_OP_AVG &&
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
    ucc_memory_type_t      mem_type   = GET_MT(args);
    ucc_datatype_t         dt         = GET_DT(args);
    ucc_rank_t             rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             size       = UCC_TL_TEAM_SIZE(team);
    size_t                 dt_size    = ucc_dt_size(dt);
    size_t                 count      = GET_COUNT(args, size);
    void                  *data_buf   = GET_DATA_BUF(args);
    void                  *dst        = task->reduce_scatter_kn.dest;
    void                  *sbuf, *rbuf;
    ptrdiff_t              peer_seg_offset, local_seg_offset;
    ucc_rank_t             peer;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 peer_seg_count, local_seg_count, local_count;

    UCC_KN_GOTO_PHASE(task->reduce_scatter_kn.phase);
    if (KN_NODE_EXTRA == node_type) {
        peer = ucc_knomial_pattern_get_proxy(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(data_buf, count * dt_size, mem_type, peer, team, task),
            task, out);
        if (p->type != KN_PATTERN_REDUCE_SCATTERX) {
            local_count = args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV ?
                ucc_coll_args_get_count(args, args->dst.info_v.counts, rank) :
                count / size;
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(dst, local_count * dt_size, mem_type, peer, team, task),
                task, out);
        }
    }

    if (KN_NODE_PROXY == node_type) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(task->reduce_scatter_kn.proxy_buf[0],
                               count * dt_size, mem_type, peer, team, task),
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
            if (UCC_OK != (status = ucc_dt_reduce(data_buf, task->reduce_scatter_kn.proxy_buf[0],
                                                  task->reduce_scatter_kn.proxy_buf[1], count,
                                                  dt, mem_type, args))) {
                tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
                task->super.super.status = status;
                return status;
            }
        }
    }

    while (!ucc_knomial_pattern_loop_done(p)) {
        sbuf = task->reduce_scatter_kn.loop_sbuf[!ucc_knomial_pattern_loop_first_iteration(p)];

        for (loop_step = 1; loop_step < radix; loop_step++) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, loop_step);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            ucc_kn_rs_pattern_peer_seg(peer, p, &peer_seg_count, &peer_seg_offset);

            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
        }
        ucc_kn_rs_pattern_peer_seg(rank, p, &local_seg_count, &local_seg_offset);

        rbuf = task->reduce_scatter_kn.loop_rbuf[!ucc_knomial_pattern_loop_first_iteration(p)];
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
        ucc_kn_rs_pattern_next_iter(p);
    }

    local_seg_count = 0;
    if (args->coll_type == UCC_COLL_TYPE_ALLREDUCE &&
        !task->reduce_scatter_kn.scratch_mc_header) {
        /* Allreduce special  case */
        ucc_kn_rsx_pattern_dst(size, rank, radix, count, &local_seg_offset, &local_seg_count);
    }

    if (args->coll_type != UCC_COLL_TYPE_ALLREDUCE &&
        p->node_type == KN_NODE_PROXY) {
        local_seg_offset = 0;
        if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) {
            local_seg_count = ucc_coll_args_get_count(args, args->dst.info_v.counts, rank);
        } else {
            local_seg_count = count / size;
        }
    }

    if (local_seg_count) {
        status = ucc_mc_memcpy(PTR_OFFSET(dst, local_seg_offset * dt_size),
                               task->reduce_scatter_kn.loop_sbuf[1],
                               local_seg_count * dt_size, mem_type, mem_type);

        if (UCC_OK != status) {
            return status;
        }
    }

    if (KN_NODE_PROXY == node_type &&
        (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER ||
         args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV)) {
        peer = ucc_knomial_pattern_get_extra(p, rank);
        ucc_kn_rs_pattern_extra_seg(p, &peer_seg_count, &peer_seg_offset);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(task->reduce_scatter_kn.loop_sbuf[1],
                                                    peer_seg_offset * dt_size),
                                         peer_seg_count * dt_size, mem_type, peer, team, task),
                      task, out);
    } else {
        goto out;
    }

UCC_KN_PHASE_PROXY: /* unused label */
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return task->super.super.status;
    }

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_done",
                                     0);
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

static inline void setup_work_buffers_allreduce(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    int no_scratch = UCC_TL_UCP_TEAM_LIB(team)->cfg.allreduce_sra_kn_no_scratch &&
        !UCC_IS_INPLACE(*args) && KN_NODE_PROXY != task->reduce_scatter_kn.p.node_type;
    void *data_buf = UCC_IS_INPLACE(*args) ?
        args->dst.info.buffer : args->src.info.buffer;
    void *dest = args->dst.info.buffer;
    size_t max_seg = ucc_kn_rs_max_seg_count(&task->reduce_scatter_kn.p);
    size_t             dt_size    = ucc_dt_size(GET_DT(args));
    size_t             count      = GET_COUNT(args, UCC_TL_TEAM_SIZE(team));
    size_t             data_size  = count * dt_size;
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;

    task->reduce_scatter_kn.loop_sbuf[0] = data_buf;
    task->reduce_scatter_kn.loop_sbuf[1] = dest;

    if (no_scratch && (max_seg * (radix - 1) <= data_size)) {
        task->reduce_scatter_kn.loop_rbuf[0] = dest;
        task->reduce_scatter_kn.loop_rbuf[1] = PTR_OFFSET(dest, max_seg * dt_size);
    } else if (task->reduce_scatter_kn.p.node_type != KN_NODE_EXTRA) {
        ucc_assert(max_seg * radix * dt_size >= data_size);
        task->reduce_scatter_kn.proxy_buf[0] = task->reduce_scatter_kn.scratch_mc_header->addr;
        task->reduce_scatter_kn.proxy_buf[1] = dest;
        task->reduce_scatter_kn.loop_rbuf[0] = task->reduce_scatter_kn.scratch_mc_header->addr;
        task->reduce_scatter_kn.loop_rbuf[1] = task->reduce_scatter_kn.scratch_mc_header->addr;
        if (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) {
            task->reduce_scatter_kn.loop_sbuf[0] = dest;
        }
    }
}

static inline void setup_work_buffers_reduce_scatter(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    size_t             dt_size    = ucc_dt_size(GET_DT(args));
    size_t max_seg = ucc_kn_rs_max_seg_count(&task->reduce_scatter_kn.p);
    void *data_buf;
    void *dest;
    ucc_kn_radix_t         radix = task->reduce_scatter_kn.p.radix;

    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        data_buf = UCC_IS_INPLACE(*args) ?
            args->dst.info.buffer : args->src.info.buffer;
        dest = args->dst.info.buffer;
    } else {
        ucc_assert(args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV);
        data_buf = UCC_IS_INPLACE(*args) ?
            args->dst.info_v.buffer : args->src.info_v.buffer;
        dest = args->dst.info_v.buffer;
    }

    if (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) {
        if (UCC_IS_INPLACE(*args)) {
            task->reduce_scatter_kn.proxy_buf[0] = task->reduce_scatter_kn.scratch_mc_header->addr;
            task->reduce_scatter_kn.proxy_buf[1] = dest;

            task->reduce_scatter_kn.loop_sbuf[0] = dest;
            task->reduce_scatter_kn.loop_sbuf[1] = task->reduce_scatter_kn.scratch_mc_header->addr;
            task->reduce_scatter_kn.loop_rbuf[0] = PTR_OFFSET(task->reduce_scatter_kn.scratch_mc_header->addr,
                                                              max_seg * dt_size);
            task->reduce_scatter_kn.loop_rbuf[1] = PTR_OFFSET(task->reduce_scatter_kn.scratch_mc_header->addr,
                                                              max_seg * dt_size);
        } else {
            task->reduce_scatter_kn.proxy_buf[0] = PTR_OFFSET(task->reduce_scatter_kn.scratch_mc_header->addr,
                                                              max_seg * (radix - 1) * dt_size);
            task->reduce_scatter_kn.proxy_buf[1] = task->reduce_scatter_kn.proxy_buf[0];

            task->reduce_scatter_kn.loop_sbuf[0] = task->reduce_scatter_kn.proxy_buf[0];

            task->reduce_scatter_kn.loop_sbuf[1] = task->reduce_scatter_kn.proxy_buf[0];
            task->reduce_scatter_kn.loop_rbuf[0] = task->reduce_scatter_kn.scratch_mc_header->addr;
            task->reduce_scatter_kn.loop_rbuf[1] = task->reduce_scatter_kn.scratch_mc_header->addr;
        }
    } else if (KN_NODE_EXTRA != task->reduce_scatter_kn.p.node_type) {
        task->reduce_scatter_kn.loop_sbuf[0] = data_buf;
        task->reduce_scatter_kn.loop_sbuf[1] = task->reduce_scatter_kn.scratch_mc_header->addr;
        task->reduce_scatter_kn.loop_rbuf[0] = PTR_OFFSET(task->reduce_scatter_kn.scratch_mc_header->addr,
                                                          max_seg * dt_size);
        task->reduce_scatter_kn.loop_rbuf[1] = PTR_OFFSET(task->reduce_scatter_kn.scratch_mc_header->addr,
                                                          max_seg * dt_size);
    }
}

static inline void setup_work_buffers(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t   *args = &TASK_ARGS(task);
    if (args->coll_type == UCC_COLL_TYPE_ALLREDUCE) {
        return setup_work_buffers_allreduce(task);
    } else {
        return setup_work_buffers_reduce_scatter(task);
    }
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
    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) {
        ucc_kn_rsv_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                                args->dst.info_v.counts,
                                UCC_COLL_ARGS64(args), &task->reduce_scatter_kn.p);
        task->reduce_scatter_kn.dest = args->dst.info_v.buffer;
    } else {
        task->reduce_scatter_kn.dest = args->dst.info.buffer;
        if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
            ucc_kn_rs_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                                   GET_COUNT(args, size), &task->reduce_scatter_kn.p);
            if (UCC_IS_INPLACE(*args)) {
                task->reduce_scatter_kn.dest =
                    PTR_OFFSET(args->dst.info.buffer, args->dst.info.count / size * rank *
                                                   ucc_dt_size(args->dst.info.datatype));
            }
        } else {
            ucc_assert(args->coll_type == UCC_COLL_TYPE_ALLREDUCE);
            ucc_kn_rsx_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                                   GET_COUNT(args, size), &task->reduce_scatter_kn.p);
        }

    }
    setup_work_buffers(task);
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

static inline size_t compute_scratch_size(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    size_t             dt_size    = ucc_dt_size(GET_DT(args));
    size_t             count      = GET_COUNT(args, UCC_TL_TEAM_SIZE(team));
    size_t             data_size  = count * dt_size;
    ucc_knomial_pattern_t *p      = &task->reduce_scatter_kn.p;
    ucc_kn_radix_t         radix = p->radix;
    size_t             max_seg;
    int                no_scratch;

    if (args->coll_type == UCC_COLL_TYPE_ALLREDUCE) {
        no_scratch = UCC_TL_UCP_TEAM_LIB(team)->cfg.allreduce_sra_kn_no_scratch &&
            !UCC_IS_INPLACE(*args) && KN_NODE_PROXY != p->node_type;
        max_seg = ucc_kn_rs_max_seg_count(p);
        if (no_scratch && (max_seg * (radix - 1) <= data_size)) {
            return 0;
        } else if (p->node_type != KN_NODE_EXTRA) {
            ucc_assert(max_seg * radix * dt_size >= data_size);
            return max_seg * radix * dt_size;
        }
    } else {
        max_seg = ucc_kn_rs_max_seg_count(p);
        if (KN_NODE_PROXY == p->node_type) {
            if (UCC_IS_INPLACE(*args)) {
                return max_seg * radix * dt_size;
            } else {
                return (max_seg * (radix - 1) + count) * dt_size;
            }
        } else if (KN_NODE_EXTRA != p->node_type) {
            return max_seg * radix * dt_size;
        }
    }
    return 0;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         rank      = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t         size      = UCC_TL_TEAM_SIZE(tl_team);
    ucc_memory_type_t  mem_type   = GET_MT(&coll_args->args);
    size_t             count      = GET_COUNT(&coll_args->args, size);
    ucc_knomial_pattern_t p;
    ucc_tl_ucp_task_t    *task;
    ucc_status_t          status;
    size_t                scratch_size;

    if (coll_args->args.coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) {
        ucc_kn_rsv_pattern_init(size, rank, radix,
                                coll_args->args.dst.info_v.counts,
                                UCC_COLL_ARGS64(&coll_args->args), &p);
    } else {
        if (coll_args->args.coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
            ucc_kn_rs_pattern_init(size, rank, radix, count, &p);
        } else {
            ucc_assert(coll_args->args.coll_type == UCC_COLL_TYPE_ALLREDUCE);
            ucc_kn_rsx_pattern_init(size, rank, radix, count, &p);
        }
    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    task->reduce_scatter_kn.scratch_mc_header = NULL;
    task->reduce_scatter_kn.p                 = p;

    scratch_size = compute_scratch_size(task);
    if (scratch_size) {
        status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch_mc_header,
                              scratch_size, mem_type);
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
    size_t             count   = GET_COUNT(&coll_args->args, size);
    ucc_kn_radix_t     radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatter_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix, size, count);
    return ucc_tl_ucp_reduce_scatter_knomial_init_r(coll_args, team, task_h,
                                                    radix);
}
