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

#define CONGESTION_THRESHOLD    8

void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *ctask);

ucc_status_t ucc_tl_ucp_alltoall_onesided_sched_start(ucc_coll_task_t *ctask)
{
    return ucc_schedule_start(ctask);
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_sched_finalize(ucc_coll_task_t *ctask)
{
    ucc_schedule_t *schedule = ucc_derived_of(ctask, ucc_schedule_t);
    ucc_status_t status;
    status = ucc_schedule_finalize(ctask);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task   = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team   = TASK_TEAM(task);
    ptrdiff_t          src    = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest   = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    size_t             nelems = TASK_ARGS(task).src.info.count;
    ucc_rank_t         grank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize  = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         start  = (grank + 1) % gsize;
    ucc_rank_t         peer;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    nelems = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    dest   = dest + grank * nelems;
    for (peer = start; task->onesided.put_posted < gsize; peer = (peer + 1) % gsize) {
        UCPCHECK_GOTO(ucc_tl_ucp_put_nb((void *)(src + peer * nelems),
                                        (void *)dest, nelems, peer, team, task),
                      task, out);
    }
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return task->super.status;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_ca_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
    size_t             nreqs     = task->alltoall_onesided.tokens;
    int                start     = (grank + 1) % gsize;
    int                iteration = 0;
    size_t             count     = 0;
    int                polls     = 0;
    size_t             npolls    = task->n_polls;
    ucc_rank_t         peer;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    count = TASK_ARGS(task).src.info.count / gsize;
    count = count * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    for (peer = start; task->onesided.get_posted < gsize;
         peer = (peer + 1) % gsize) {

        UCPCHECK_GOTO(ucc_tl_ucp_get_nb(PTR_OFFSET(dest, peer * count),
                                        PTR_OFFSET(src, grank * count),
                                        count, peer, team, task),
                      task, out);
        ++iteration;

        if ((task->onesided.get_posted - task->onesided.get_completed) >= nreqs) {
            while (polls++ < npolls) {
                ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
                if ((task->onesided.get_posted - task->onesided.get_completed) < nreqs) {
                    break;
                }
            }
            if (polls >= npolls) {
                break;
            }
        }
    }
    task->alltoall_onesided.iteration = iteration;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return task->super.status;
}


void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    int                polls = 0;

    if (!UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
        while (polls++ < task->n_polls) {
            ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                goto complete;
            }
        }
        return;
    }
complete:
    task->super.status = UCC_OK;
}

void ucc_tl_ucp_alltoall_onesided_ca_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         start     = (grank + 1) % gsize;
    ucc_rank_t         peer      = (start + task->alltoall_onesided.iteration) % gsize;
    int                iteration = task->alltoall_onesided.iteration;
    size_t             nreqs     = task->alltoall_onesided.tokens;
    int                polls     = 0;
    int64_t            npolls;
    size_t             count;

    count = TASK_ARGS(task).src.info.count / gsize;
    count = count * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    npolls = count - nreqs;
    if (npolls < task->n_polls) {
        npolls = task->n_polls;
    }

    for (; task->onesided.get_posted < gsize; peer = (peer + 1) % gsize) {
        UCPCHECK_GOTO(ucc_tl_ucp_get_nb(PTR_OFFSET(dest, peer * count),
                                        PTR_OFFSET(src, grank * count),
                                        count, peer, team, task),
                      task, out);
        ++iteration;

        if ((task->onesided.get_posted - task->onesided.get_completed) >= nreqs) {
            while (polls++ < npolls) {
                ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
                if ((task->onesided.get_posted - task->onesided.get_completed) < nreqs) {
                    break;
                }
            }
            if (polls >= npolls) {
                task->alltoall_onesided.iteration = iteration;
                return;
            }
        }
    }
    if (!UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
        while (polls++ < task->n_polls) {
            ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                goto complete;
            }
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
        if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
            goto complete;
        }
        return;
    }

complete:
    task->super.status = UCC_OK;
out:
    return;
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

ucc_status_t ucc_tl_ucp_alltoall_onesided_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h)
{
    ucc_schedule_t              *schedule = NULL;
    ucc_tl_ucp_team_t           *tl_team =
        ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t         barrier_coll_args = {
        .team = team->params.team,
        .args.coll_type = UCC_COLL_TYPE_BARRIER,
    };
    size_t                       perc_bw =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_percent_bw;
    ucc_coll_task_t             *barrier_task;
    ucc_coll_task_t             *a2a_task;
    ucc_tl_ucp_task_t           *task;
    ucc_status_t                 status;
    ucc_tl_ucp_schedule_t       *tl_schedule;
    size_t                       nelems;
    size_t                       rate;
    size_t                       ratio;
    ucp_ep_h                     ep;
    ucp_ep_evaluate_perf_param_t param;
    ucp_ep_evaluate_perf_attr_t  attr;
    ucc_sbgp_t                  *sbgp;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args, (ucc_tl_ucp_schedule_t **)&tl_schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    schedule = &tl_schedule->super.super;
    ucc_schedule_init(schedule, coll_args, team);
    schedule->super.post     = ucc_tl_ucp_alltoall_onesided_sched_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_alltoall_onesided_sched_finalize;

    sbgp                 = ucc_topo_get_sbgp(tl_team->topo, UCC_SBGP_NODE);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    if (sbgp->group_size < CONGESTION_THRESHOLD) {
        task->super.post     = ucc_tl_ucp_alltoall_onesided_start;
        task->super.progress = ucc_tl_ucp_alltoall_onesided_progress;
    } else {
        task->super.post     = ucc_tl_ucp_alltoall_onesided_ca_start;
        task->super.progress = ucc_tl_ucp_alltoall_onesided_ca_progress;
    }
    task->super.finalize = ucc_tl_ucp_alltoall_onesided_finalize;
    a2a_task = &task->super;

    status = ucc_tl_ucp_coll_init(&barrier_coll_args, team, &barrier_task);
    if (status != UCC_OK) {
        return status;
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
    param.message_size = (1 << 20);
    ucc_tl_ucp_get_ep(tl_team,
        (UCC_TL_TEAM_RANK(tl_team) + 1) % UCC_TL_TEAM_SIZE(tl_team), &ep);
    ucp_ep_evaluate_perf(ep, &param, &attr);

    rate  = (param.message_size / attr.estimated_time) / (param.message_size);
    rate  = rate * (double)(perc_bw / 100.0);
    ratio = nelems * sbgp->group_size;
    task->alltoall_onesided.tokens = rate / ratio;
    if (task->alltoall_onesided.tokens < 1) {
        task->alltoall_onesided.tokens = 1;
    }
    task->alltoall_onesided.iteration = 0;

    ucc_schedule_add_task(schedule, a2a_task);
    ucc_task_subscribe_dep(&schedule->super, a2a_task, UCC_EVENT_SCHEDULE_STARTED);

    ucc_schedule_add_task(schedule, barrier_task);
    ucc_task_subscribe_dep(a2a_task, barrier_task,
                           UCC_EVENT_COMPLETED);
    *task_h = &schedule->super;
out:
    return status;
}
