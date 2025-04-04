/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatter.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_dt_reduce.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->reduce_scatter_kn.phase = _phase;                                \
    } while (0)

#define GET_COUNT(_args)                                                       \
    ({                                                                         \
        size_t _count = 0;                                                     \
        switch ((_args)->coll_type) {                                          \
        case UCC_COLL_TYPE_ALLREDUCE:                                          \
        case UCC_COLL_TYPE_REDUCE:                                             \
            _count = (_args)->dst.info.count;                                  \
            break;                                                             \
        case UCC_COLL_TYPE_REDUCE_SCATTER:                                     \
            _count = UCC_IS_INPLACE(*(_args))                                  \
                         ? (_args)->dst.info.count                             \
                         : (_args)->src.info.count;                            \
            break;                                                             \
        default:                                                               \
            break;                                                             \
        }                                                                      \
        _count;                                                                \
    })

#define GET_DT(_args)                                                          \
    ((_args)->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV)                      \
        ? (_args)->dst.info_v.datatype                                         \
        : (_args)->dst.info.datatype

typedef struct ucc_tl_ucp_rs_work_buf {
    void *src_data;
    void *dst_data;
    void *src_loop;
    void *dst_loop;
    void *dst_proxy;
    void *reduce_proxy;
    void *reduce_loop;
} ucc_tl_ucp_rs_work_buf_t;

/* get work buffers for allreduce and reduce */
static inline void get_sbuf_rbuf_ar(ucc_tl_ucp_task_t *task,
                                    size_t block_count,
                                    ucc_tl_ucp_rs_work_buf_t *wb)
{
    ucc_coll_args_t       *args      = &TASK_ARGS(task);
    size_t                 dt_size   = ucc_dt_size(args->dst.info.datatype);
    void                  *scratch   = task->reduce_scatter_kn.scratch;
    ucc_knomial_pattern_t *p         = &task->reduce_scatter_kn.p;
    size_t offset, local_seg_count;
    ptrdiff_t local_seg_offset;
    void *sbuf, *rbuf;

    if (ucc_knomial_pattern_loop_first_iteration(p)) {
        sbuf = ((KN_NODE_PROXY ==  p->node_type) || UCC_IS_INPLACE(*args))
                ? args->dst.info.buffer: args->src.info.buffer;
        rbuf = scratch;
    } else {
        sbuf = scratch;
        if (!ucc_knomial_pattern_loop_last_iteration(p) ||
            (task->reduce_scatter_kn.scratch_mc_header != NULL)) {
            rbuf = PTR_OFFSET(sbuf, block_count * dt_size);
        } else {
            ucc_sra_kn_get_offset_and_seglen(args->dst.info.count, dt_size,
                                             p->rank, p->size,
                                             p->radix, &local_seg_offset,
                                             &local_seg_count);
            local_seg_offset = local_seg_offset / dt_size;
            if ((local_seg_offset <= block_count) || (local_seg_count == 0)) {
                rbuf = PTR_OFFSET(sbuf, block_count * dt_size);
            } else {
                offset = (local_seg_offset - block_count) % local_seg_count;
                /* check we have enough space to store segments */
                ucc_assert(args->dst.info.count - (block_count + offset) >=
                           local_seg_count * (ucc_kn_compute_step_radix(p) - 1));
                rbuf = PTR_OFFSET(sbuf, (block_count + offset) * dt_size);
            }
        }
    }

    if (ucc_knomial_pattern_loop_last_iteration(p)) {
        ucc_sra_kn_get_offset_and_seglen(args->dst.info.count, dt_size, p->rank,
                                         p->size, p->radix, &local_seg_offset,
                                         &local_seg_count);
        wb->reduce_loop = PTR_OFFSET(args->dst.info.buffer, local_seg_offset);
    } else {
        wb->reduce_loop = scratch;
    }
    wb->src_data     = UCC_IS_INPLACE(*args) ? args->dst.info.buffer
                                             : args->src.info.buffer;
    wb->src_loop     = sbuf;
    wb->dst_loop     = rbuf;
    wb->dst_proxy    = scratch;
    wb->reduce_proxy = args->dst.info.buffer;
}

/* get work buffers for reduce scatter */
static inline void get_sbuf_rbuf_rs(ucc_tl_ucp_task_t *task,
                                    ucc_tl_ucp_rs_work_buf_t *wb)
{
    ucc_coll_args_t       *args    = &TASK_ARGS(task);
    ucc_knomial_pattern_t *p       = &task->reduce_scatter_kn.p;
    void                  *scratch = task->reduce_scatter_kn.scratch;
    size_t                 dt_size = ucc_dt_size(args->dst.info.datatype);
    ucc_rank_t             trank   = task->subset.myrank;
    ucc_rank_t             tsize   = task->subset.map.ep_num;
    ucc_kn_radix_t         radix   = p->radix;
    size_t max_seg;
    void *sbuf, *rbuf, *data_buf;

    max_seg = task->reduce_scatter_kn.max_seg;

    if (UCC_IS_INPLACE(*args)) {
        data_buf     = args->dst.info.buffer;
        wb->dst_data = PTR_OFFSET(args->dst.info.buffer,
                                  (args->dst.info.count / tsize) *
                                  trank * dt_size);
    } else {
        data_buf     = args->src.info.buffer;
        wb->dst_data = args->dst.info.buffer;
    }

    if (KN_NODE_PROXY == p->node_type) {
        if (UCC_IS_INPLACE(*args)) {
            rbuf = PTR_OFFSET(scratch, max_seg * dt_size);
            if (ucc_knomial_pattern_loop_first_iteration(p)) {
                sbuf = args->dst.info.buffer;
            } else {
                sbuf = scratch;
            }
            wb->dst_proxy = scratch;
            wb->reduce_proxy = args->dst.info.buffer;
        } else {
            sbuf = PTR_OFFSET(scratch, max_seg * (radix - 1) * dt_size);
            rbuf = scratch;
            wb->dst_proxy    = PTR_OFFSET(scratch,
                                          max_seg * (radix - 1) * dt_size);
            wb->reduce_proxy = wb->dst_proxy;
        }
    } else {
        rbuf = PTR_OFFSET(scratch, max_seg * dt_size);
        if (ucc_knomial_pattern_loop_first_iteration(p)) {
            sbuf = data_buf;
        } else {
            sbuf = scratch;
        }
    }

    if (ucc_knomial_pattern_loop_last_iteration(p)) {
         if (KN_NODE_PROXY == p->node_type) {
            wb->reduce_loop = PTR_OFFSET(scratch,
                                         max_seg * (radix - 1) * dt_size);
        } else {
            wb->reduce_loop = wb->dst_data;
        }
    } else {
        if (KN_NODE_PROXY == p->node_type) {
            if (UCC_IS_INPLACE(*args)) {
                wb->reduce_loop = scratch;
            } else {
                wb->reduce_loop = PTR_OFFSET(scratch, max_seg * (radix - 1) * dt_size);
            }
        } else {
            wb->reduce_loop = scratch;
        }
    }

    wb->src_loop = sbuf;
    wb->dst_loop = rbuf;
    wb->src_data = data_buf;
}

static inline void get_rs_work_buf(ucc_tl_ucp_task_t *task,
                                   size_t block_count,
                                   ucc_tl_ucp_rs_work_buf_t *wb)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    switch (args->coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_REDUCE:
        return get_sbuf_rbuf_ar(task, block_count, wb);
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return get_sbuf_rbuf_rs(task, wb);
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
    default:
        ucc_assert(0);
        return;
    }
}

/* return the rank of the peer for the given rank and pattern
   taken into account the root and map */
static inline ucc_rank_t get_physical_rank(ucc_tl_ucp_task_t *task, ucc_rank_t rank,
                                           ucc_rank_t root, ucc_rank_t size)
{
    return INV_VRANK(ucc_ep_map_eval(task->subset.map, rank), root, size);
}

void ucc_tl_ucp_reduce_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t        *task            = ucc_derived_of(coll_task,
                                                               ucc_tl_ucp_task_t);
    ucc_coll_args_t          *args            = &TASK_ARGS(task);
    ucc_tl_ucp_team_t        *team            = TASK_TEAM(task);
    ucc_knomial_pattern_t    *p               = &task->reduce_scatter_kn.p;
    ucc_kn_radix_t            radix           = p->radix;
    uint8_t                   node_type       = p->node_type;
    ucc_memory_type_t         mem_type        = args->dst.info.mem_type;
    size_t                    count           = GET_COUNT(args);
    ucc_datatype_t            dt              = GET_DT(args);
    size_t                    dt_size         = ucc_dt_size(dt);
    size_t                    data_size       = count * dt_size;
    ucc_rank_t                rank            = task->subset.myrank;
    ucc_rank_t                size            = task->subset.map.ep_num;
    ucc_rank_t                root            = 0;
    size_t                    local_seg_count = 0;
    ucc_tl_ucp_rs_work_buf_t  wb              = (ucc_tl_ucp_rs_work_buf_t){0};
    ptrdiff_t                peer_seg_offset, local_seg_offset;
    ucc_rank_t               peer;
    ucc_status_t             status;
    ucc_kn_radix_t           step_radix, loop_step;
    size_t                   block_count, peer_seg_count;
    void                    *local_data;
    int                      is_avg;

    if (args->coll_type == UCC_COLL_TYPE_REDUCE) {
        root = args->root;
        rank = VRANK(rank, root, size);
    }

    UCC_KN_REDUCE_GOTO_PHASE(task->reduce_scatter_kn.phase);
    block_count = ucc_sra_kn_compute_block_count(count, rank, p);
    get_rs_work_buf(task, block_count, &wb);
    if (KN_NODE_EXTRA == node_type) {
        peer = get_physical_rank(task, ucc_knomial_pattern_get_proxy(p, rank),
                                 root, size);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(wb.src_data, data_size, mem_type,
                                         peer, team, task),
                      task, out);
        if (p->type != KN_PATTERN_REDUCE_SCATTERX) {
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(wb.dst_data, (count / size) * dt_size,
                                             mem_type, peer, team, task),
                          task, out);
        }
    }

    if (KN_NODE_PROXY == node_type) {
        peer = get_physical_rank(task, ucc_knomial_pattern_get_extra(p, rank),
                                 root, size);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(wb.dst_proxy, data_size, mem_type,
                                         peer, team, task),
                      task, out);
    }

UCC_KN_PHASE_EXTRA:
    block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
    get_rs_work_buf(task, block_count, &wb);
    if ((KN_NODE_PROXY == node_type) || (KN_NODE_EXTRA == node_type)) {
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_EXTRA);
            return;
        }
        if (KN_NODE_EXTRA == node_type) {
            goto out;
        }
        status = ucc_dt_reduce(wb.src_data, wb.dst_proxy, wb.reduce_proxy,
                               count, dt, args, 0, 0,
                               task->reduce_scatter_kn.executor,
                               &task->reduce_scatter_kn.etask);
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
    while (!ucc_knomial_pattern_loop_done(p)) {
        block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
        local_seg_count = 0;
        ucc_kn_rs_pattern_peer_seg(rank, p, &local_seg_count,
                                   &local_seg_offset);
        get_rs_work_buf(task, block_count, &wb);
        for (loop_step = radix - 1; loop_step > 0; loop_step--) {
            peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
            if (peer == UCC_KN_PEER_NULL) {
                continue;
            }
            ucc_kn_rs_pattern_peer_seg(peer, p, &peer_seg_count,
                                       &peer_seg_offset);
            peer = get_physical_rank(task, peer, root, size);
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(PTR_OFFSET(wb.src_loop, peer_seg_offset * dt_size),
                                   peer_seg_count * dt_size, mem_type, peer,
                                   team, task),
                task, out);
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(wb.dst_loop, local_seg_count * dt_size, mem_type,
                                   peer, team, task),
                task, out);
            wb.dst_loop = PTR_OFFSET(wb.dst_loop, local_seg_count * dt_size);
        }

UCC_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_KN_PHASE_LOOP);
            return;
        }
        if (task->tagged.send_posted > p->iteration * (radix - 1)) {
            step_radix      = ucc_kn_compute_step_radix(p);
            local_seg_count = 0;
            block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
            ucc_kn_rs_pattern_peer_seg(rank, p, &local_seg_count,
                                       &local_seg_offset);
            get_rs_work_buf(task, block_count, &wb);
            local_data  = PTR_OFFSET(wb.src_loop, local_seg_offset * dt_size);
            is_avg      = (args->op == UCC_OP_AVG) &&
                          (UCC_TL_UCP_TEAM_LIB(team)->cfg.reduce_avg_pre_op ?
                                   ucc_knomial_pattern_loop_first_iteration(p) :
                                   ucc_knomial_pattern_loop_last_iteration(p));
            status = ucc_dt_reduce_strided(local_data, wb.dst_loop, wb.reduce_loop,
                                           step_radix - 1, local_seg_count,
                                           local_seg_count * dt_size, dt, args,
                                           is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                                           AVG_ALPHA(task), task->reduce_scatter_kn.executor,
                                           &task->reduce_scatter_kn.etask);
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
        if ((args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) &&
            (KN_NODE_PROXY == node_type) &&
            ucc_knomial_pattern_loop_last_iteration(p)) {
            get_rs_work_buf(task, 0, &wb);
            peer            = ucc_knomial_pattern_get_extra(p, rank);
            peer_seg_count  = 0;
            peer_seg_offset = 0;
            ucc_kn_rs_pattern_extra_seg(p, &peer_seg_count, &peer_seg_offset);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(
                            PTR_OFFSET(wb.reduce_loop,
                                       peer_seg_offset * dt_size),
                            peer_seg_count * dt_size, mem_type, peer, team, task),
                          task, out);
            ucc_mc_memcpy(wb.dst_data, wb.reduce_loop, peer_seg_count * dt_size,
                          mem_type, mem_type);
        }
        ucc_kn_rs_pattern_next_iter(p);
    }

UCC_KN_PHASE_COMPLETE:
UCC_KN_PHASE_PROXY:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        SAVE_STATE(UCC_KN_PHASE_PROXY);
        return;
    }
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_done",
                                     0);
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         size  = task->subset.map.ep_num;
    ucc_coll_type_t    ct    = args->coll_type;
    ucc_rank_t         root  = (ct == UCC_COLL_TYPE_REDUCE) ? args->root : 0;
    ucc_rank_t         rank  = VRANK(task->subset.myrank, root, size);
    size_t             count = GET_COUNT(args);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_reduce_scatter_kn_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if ((ct == UCC_COLL_TYPE_ALLREDUCE) ||
        (ct == UCC_COLL_TYPE_REDUCE)) {
        ucc_kn_rsx_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                                count, &task->reduce_scatter_kn.p);
    } else {
        ucc_kn_rs_pattern_init(size, rank, task->reduce_scatter_kn.p.radix,
                               count, &task->reduce_scatter_kn.p);
    }

    if (!task->reduce_scatter_kn.scratch_mc_header) {
        task->reduce_scatter_kn.scratch = args->dst.info.buffer;
    }
    task->reduce_scatter_kn.phase = UCC_KN_PHASE_INIT;

    status = ucc_coll_task_get_executor(&task->super,
                                        &task->reduce_scatter_kn.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

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

static size_t compute_scratch_size(ucc_tl_ucp_task_t *task)
{
    ucc_coll_args_t      *args      = &TASK_ARGS(task);
    ucc_base_coll_args_t *coll_args = &task->super.bargs;
    size_t                count     = GET_COUNT(args);
    size_t                dt_size   = ucc_dt_size(GET_DT(args));
    size_t                max_seg   = task->reduce_scatter_kn.max_seg;
    size_t data_size;
    ucc_kn_radix_t step_radix;
    size_t max_recv_size;

    if ((args->coll_type == UCC_COLL_TYPE_ALLREDUCE) ||
        (args->coll_type == UCC_COLL_TYPE_REDUCE)) {
        if (KN_NODE_EXTRA != task->reduce_scatter_kn.p.node_type) {
            if (coll_args->mask & UCC_BASE_CARGS_MAX_FRAG_COUNT) {
                count = coll_args->max_frag_count;
            }
            data_size     = count * dt_size;
            step_radix    = ucc_kn_compute_step_radix(&task->reduce_scatter_kn.p);
            max_recv_size = ucc_sra_kn_compute_seg_size(count, step_radix, 0) *
                            step_radix * dt_size;
            if (UCC_IS_INPLACE(coll_args->args) ||
                (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) ||
                max_recv_size > data_size) {
                return ucc_max(max_recv_size, data_size);
            } else {
                return 0;
            }
        }
    } else {
        step_radix = task->reduce_scatter_kn.p.radix;
        if (KN_NODE_PROXY == task->reduce_scatter_kn.p.node_type) {
            if (UCC_IS_INPLACE(*args)) {
                return max_seg * step_radix * dt_size;
            } else {
                return (max_seg * (step_radix - 1) + count) * dt_size;
            }
        } else if (KN_NODE_EXTRA != task->reduce_scatter_kn.p.node_type) {
            return max_seg * step_radix * dt_size;
        }
    }
    return 0;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_init_r(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_coll_task_t **task_h,
                                         ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_memory_type_t  mem_type  = coll_args->args.dst.info.mem_type;
    size_t             count     = GET_COUNT(&coll_args->args);
    ucc_coll_type_t    ct        = coll_args->args.coll_type;
    ucc_sbgp_t        *sbgp;
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    size_t             scratch_size;
    ucc_rank_t         rank, size;
    ptrdiff_t          max_seg_offset;

    if (ucc_unlikely(!UCC_IS_INPLACE(coll_args->args) &&
                     (coll_args->args.src.info.mem_type !=
                      coll_args->args.dst.info.mem_type))) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.flags    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_ucp_reduce_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_knomial_finalize;

    if (tl_team->cfg.use_reordering && ct == UCC_COLL_TYPE_ALLREDUCE) {
        sbgp = ucc_topo_get_sbgp(tl_team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }

    rank = task->subset.myrank;
    size = task->subset.map.ep_num;

    if (ct == UCC_COLL_TYPE_ALLREDUCE) {
        ucc_kn_rsx_pattern_init(size, rank, radix,
                                count, &task->reduce_scatter_kn.p);

    } else if (ct == UCC_COLL_TYPE_REDUCE) {
        ucc_kn_rsx_pattern_init(size, VRANK(rank, coll_args->args.root, size),
                                radix, count, &task->reduce_scatter_kn.p);
    } else {
        ucc_kn_rs_pattern_init(size, rank, radix,
                               count, &task->reduce_scatter_kn.p);
    }

    ucc_kn_rs_pattern_peer_seg(0, &task->reduce_scatter_kn.p,
                               &task->reduce_scatter_kn.max_seg,
                               &max_seg_offset);
    task->reduce_scatter_kn.scratch_mc_header = NULL;

    scratch_size = compute_scratch_size(task);
    if (scratch_size != 0) {
        status = ucc_mc_alloc(&task->reduce_scatter_kn.scratch_mc_header,
                              scratch_size, mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
            ucc_tl_ucp_coll_finalize(&task->super);
            return status;
        }
        task->reduce_scatter_kn.scratch =
            task->reduce_scatter_kn.scratch_mc_header->addr;
    }

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *team,
                                       ucc_coll_task_t **task_h)
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
