/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "tl_ucp_sendrecv.h"

#define CONGESTION_THRESHOLD 8

/* Common helper function to check completion and handle polling */
static inline int alltoall_onesided_handle_completion(
    ucc_tl_ucp_task_t *task, uint32_t *posted, uint32_t *completed,
    uint32_t nreqs, int64_t npolls)
{
    int64_t polls = 0;

    if ((*posted - *completed) >= nreqs) {
        while (polls < npolls) {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            ++polls;
            if ((*posted - *completed) < nreqs) {
                break;
            }
        }
        if (polls >= npolls) {
            return 0; /* Return 0 to indicate should return */
        }
    }
    return 1; /* Return 1 to indicate should continue */
}

/* Common helper function to wait for all operations to complete */
static inline void alltoall_onesided_wait_completion(ucc_tl_ucp_task_t *task,
                                                     int64_t npolls)
{
    int64_t polls = 0;

    if (!UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
        while (polls++ < npolls) {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                task->super.status = UCC_OK;
                return;
            }
        }
        return;
    }
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_sched_start(ucc_coll_task_t *ctask)
{
    return ucc_schedule_start(ctask);
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_sched_finalize(ucc_coll_task_t *ctask)
{
    ucc_schedule_t *schedule = ucc_derived_of(ctask, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(ctask);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_finalize(ucc_coll_task_t *coll_task)
{
    ucc_status_t status;

    status = ucc_tl_ucp_coll_finalize(coll_task);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(coll_task), "failed to finalize collective");
    }
    return status;
}

void ucc_tl_ucp_alltoall_onesided_get_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
    uint32_t           ntokens   = task->alltoall_onesided.tokens;
    int64_t            npolls    = task->alltoall_onesided.npolls;
    /* To resolve remote virtual addresses, the dst_memh is the one that must
     * have the rkey information. For this algorithm, we need to swap the
     * src and dst handles to operate correctly */
    ucc_mem_map_mem_h *dst_memh  = TASK_ARGS(task).src_memh.global_memh;
    uint32_t          *posted    = &task->onesided.get_posted;
    uint32_t          *completed = &task->onesided.get_completed;
    ucc_rank_t         peer      = (grank + *posted + 1) % gsize;
    ucc_mem_map_mem_h  src_memh;
    size_t             nelems;

    nelems   = TASK_ARGS(task).src.info.count;
    nelems   = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    src_memh = (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)
                   ? TASK_ARGS(task).dst_memh.global_memh[grank]
                   : TASK_ARGS(task).dst_memh.local_memh;

    for (; *posted < gsize; peer = (peer + 1) % gsize) {
        UCPCHECK_GOTO(ucc_tl_ucp_get_nb(PTR_OFFSET(dest, peer * nelems),
                                        PTR_OFFSET(src, grank * nelems),
                                        nelems, peer, src_memh, dst_memh, team,
                                        task),
                      task, out);

        if (!alltoall_onesided_handle_completion(task, posted, completed,
                                                 ntokens, npolls)) {
            return;
        }
    }

    alltoall_onesided_wait_completion(task, npolls);
out:
    return;
}

void ucc_tl_ucp_alltoall_onesided_put_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
    uint32_t           ntokens   = task->alltoall_onesided.tokens;
    int64_t            npolls    = task->alltoall_onesided.npolls;
    ucc_mem_map_mem_h *dst_memh  = TASK_ARGS(task).dst_memh.global_memh;
    uint32_t          *posted    = &task->onesided.put_posted;
    uint32_t          *completed = &task->onesided.put_completed;
    ucc_rank_t         peer      = (grank + *posted + 1) % gsize;
    ucc_mem_map_mem_h  src_memh;
    size_t             nelems;

    nelems   = TASK_ARGS(task).src.info.count;
    nelems   = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    src_memh = (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL)
                   ? TASK_ARGS(task).src_memh.global_memh[grank]
                   : TASK_ARGS(task).src_memh.local_memh;

    for (; *posted < gsize; peer = (peer + 1) % gsize) {
        UCPCHECK_GOTO(
            ucc_tl_ucp_put_nb(PTR_OFFSET(src, peer * nelems),
                              PTR_OFFSET(dest, grank * nelems), nelems,
                              peer, src_memh, dst_memh, team, task),
            task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_ep_flush(peer, team, task), task, out);

        if (!alltoall_onesided_handle_completion(task, posted, completed,
                                                 ntokens, npolls)) {
            return;
        }
    }

    alltoall_onesided_wait_completion(task, npolls);
out:
    return;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h)
{
    ucc_schedule_t              *schedule = NULL;
    ucc_tl_ucp_team_t           *tl_team  =
        ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t         barrier_coll_args = {
        .team = team->params.team,
        .args.coll_type = UCC_COLL_TYPE_BARRIER,
    };
    size_t                       perc_bw     =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_percent_bw;
    ucc_tl_ucp_alltoall_onesided_alg_t alg   =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_alg;
    ucc_tl_ucp_schedule_t       *tl_schedule = NULL;
    ucc_coll_task_t             *barrier_task;
    ucc_coll_task_t             *a2a_task;
    ucc_tl_ucp_task_t           *task;
    ucc_status_t                 status;
    size_t                       nelems;
    double                       rate;
    size_t                       ratio;
    ucp_ep_h                     ep;
    ucp_ep_evaluate_perf_param_t param;
    ucp_ep_evaluate_perf_attr_t  attr;
    int64_t                      npolls;
    ucc_sbgp_t                  *sbgp;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) ||
        (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS &&
            (!(coll_args->args.flags &
               UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)))) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "non memory mapped buffers are not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        return status;
    }

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH)) {
        coll_args->args.src_memh.global_memh = NULL;
    } else {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                "onesided alltoall requires global memory handles for src buffers");
            status = UCC_ERR_INVALID_PARAM;
            return status;
        }
    }

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH)) {
        coll_args->args.dst_memh.global_memh = NULL;
    } else {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                "onesided alltoall requires global memory handles for dst buffers");
            status = UCC_ERR_INVALID_PARAM;
            return status;
        }
    }
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&tl_schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    schedule = &tl_schedule->super.super;
    ucc_schedule_init(schedule, coll_args, team);
    schedule->super.post     = ucc_tl_ucp_alltoall_onesided_sched_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_alltoall_onesided_sched_finalize;

    sbgp = ucc_topo_get_sbgp(tl_team->topo, UCC_SBGP_NODE);
    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.finalize = ucc_tl_ucp_alltoall_onesided_finalize;
    a2a_task             = &task->super;

    status = ucc_tl_ucp_coll_init(&barrier_coll_args, team, &barrier_task);
    if (status != UCC_OK) {
        goto out;
    }
    if (perc_bw > 100) {
        perc_bw = 100;
    } else if (perc_bw == 0) {
        perc_bw = 1;
    }

    nelems             = TASK_ARGS(task).src.info.count;
    nelems             = nelems / UCC_TL_TEAM_SIZE(tl_team);
    param.field_mask   = UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE;
    attr.field_mask    = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME;
    param.message_size = nelems * ucc_dt_size(TASK_ARGS(task).src.info.datatype);;
    ucc_tl_ucp_get_ep(
        tl_team, (UCC_TL_TEAM_RANK(tl_team) + 1) % UCC_TL_TEAM_SIZE(tl_team),
        &ep);
    ucp_ep_evaluate_perf(ep, &param, &attr);

    rate  = (1 / attr.estimated_time) * (double)(perc_bw / 100.0);
    ratio = (nelems > 0) ? nelems * sbgp->group_size : 1;
    task->alltoall_onesided.tokens = rate / ratio;
    if (task->alltoall_onesided.tokens < 1) {
        task->alltoall_onesided.tokens = 1;
    }
    task->super.post = ucc_tl_ucp_alltoall_onesided_start;
    npolls           = task->n_polls;
    if (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_GET ||
       (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_AUTO &&
                                    sbgp->group_size >= CONGESTION_THRESHOLD)) {
        npolls = nelems * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
        if (npolls < task->n_polls) {
            npolls = task->n_polls;
        }
        task->super.progress = ucc_tl_ucp_alltoall_onesided_get_progress;
    } else {
        task->super.progress = ucc_tl_ucp_alltoall_onesided_put_progress;
    }
    task->alltoall_onesided.npolls = npolls;

    ucc_schedule_add_task(schedule, a2a_task);
    ucc_task_subscribe_dep(&schedule->super, a2a_task,
                           UCC_EVENT_SCHEDULE_STARTED);

    ucc_schedule_add_task(schedule, barrier_task);
    ucc_task_subscribe_dep(a2a_task, barrier_task,
                           UCC_EVENT_COMPLETED);
    *task_h = &schedule->super;
    return status;
out:
    if (tl_schedule) {
        ucc_tl_ucp_put_schedule(&tl_schedule->super.super);
    }
    return status;
}
