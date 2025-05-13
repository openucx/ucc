/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"

ucc_status_t ucc_tl_ucp_allgather_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t    *team      = TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    size_t                count     = TASK_ARGS(task).dst.info.count;
    void                 *sbuf      = TASK_ARGS(task).src.info.buffer;
    void                 *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t     smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t     rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t        dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t            trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t            tsize     = UCC_TL_TEAM_SIZE(team);
    size_t                data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t          status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_linear_start",
                                     0);

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->allgather_linear.copy_task = NULL;

    /* Copy local data to the receive buffer if not in-place */
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        status = ctx->copy.post(PTR_OFFSET(rbuf, data_size * trank), rmem,
                                sbuf, smem, data_size, task,
                                &task->allgather_linear.copy_task);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

/* Get the number of requests in flight to be used for the allgather batched algorithm 
 * If the number of requests is not specified, use the number of team size - 1
 * If number of request is bigger than the team size - 1, use the team size - 1
 */
static unsigned long get_num_reqs(const ucc_tl_ucp_team_t *team)
{
    unsigned long reqs =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.allgather_batched_num_posts;
    ucc_rank_t max_req = UCC_TL_TEAM_SIZE(team) - 1;
    reqs = (reqs > max_req || reqs == UCC_ULUNITS_AUTO) ? max_req : reqs;
    return reqs;
}

ucc_status_t ucc_tl_ucp_allgather_batched_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, unsigned long nreqs)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_init_task(coll_args, team);

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        ucc_tl_ucp_put_task(task);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!UCC_IS_INPLACE(coll_args->args)) {
        if (UCC_TL_UCP_TEAM_CTX(tl_team)->cfg.local_copy_type ==
            UCC_TL_UCP_LOCAL_COPY_TYPE_EC) {
            task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
        }
    }

    task->super.post     = ucc_tl_ucp_allgather_linear_start;
    task->super.progress = ucc_tl_ucp_allgather_linear_progress;
    task->allgather_linear.nreqs =
        nreqs == 0 ? UCC_TL_TEAM_SIZE(tl_team) - 1 : nreqs;
    *task_h = &task->super;

    return UCC_OK;
}

/* Linear Batched K-send/receive in flight */
ucc_status_t
ucc_tl_ucp_allgather_linear_batched_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t      *team,
                                         ucc_coll_task_t     **task_h)
{
    return ucc_tl_ucp_allgather_batched_init(
        coll_args, team, task_h,
        get_num_reqs(ucc_derived_of(team, ucc_tl_ucp_team_t)));
}

/* Linear One-Shot version of allgather */
ucc_status_t ucc_tl_ucp_allgather_linear_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_coll_task_t     **task_h)
{
    // 0 means one-shot, team size - 1 request will be used
    return ucc_tl_ucp_allgather_batched_init(coll_args, team, task_h, 0);
}

void ucc_tl_ucp_allgather_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t    *team      = TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    ucc_rank_t            trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t            tsize     = UCC_TL_TEAM_SIZE(team);
    void                 *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t     rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t        dt        = TASK_ARGS(task).dst.info.datatype;
    size_t                count     = TASK_ARGS(task).dst.info.count;
    size_t                data_size = (count / tsize) * ucc_dt_size(dt);
    int                   nreqs     = task->allgather_linear.nreqs;
    int                   polls     = 0;
    void                 *tmpsend   = UCC_IS_INPLACE(TASK_ARGS(task))
                                          ? PTR_OFFSET(rbuf, trank * data_size)
                                          : TASK_ARGS(task).src.info.buffer;
    ucc_memory_type_t     smem      = UCC_IS_INPLACE(TASK_ARGS(task))
                                          ? rmem
                                          : TASK_ARGS(task).src.info.mem_type;
    void                 *tmprecv;
    ucc_rank_t            peer;
    ucc_status_t          status;

    while ((task->tagged.send_posted < tsize - 1 ||
            task->tagged.recv_posted < tsize - 1) &&
           (polls++ < task->n_polls)) {

        /* Progress UCP worker */
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);

        /* Try to send data to clockwise peer */
        while ((task->tagged.send_posted < tsize - 1) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer    = (trank + 1 + task->tagged.send_posted) % tsize;
            /* Send my data to peer */
            UCPCHECK_GOTO(
                ucc_tl_ucp_send_nb(tmpsend, data_size, smem, peer, team, task),
                task, err);
            polls = 0;
        }

        /* Receive peer's data from counter-clockwise peer to avoid deadlock*/
        while ((task->tagged.recv_posted < tsize - 1) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer    = (tsize + trank - 1 - task->tagged.recv_posted) % tsize;
            tmprecv = PTR_OFFSET(rbuf, peer * data_size);
            UCPCHECK_GOTO(
                ucc_tl_ucp_recv_nb(tmprecv, data_size, rmem, peer, team, task),
                task, err);
            polls = 0;
        }
    }

    if (task->tagged.send_posted < tsize - 1 ||
        task->tagged.recv_posted < tsize - 1) {
        return;
    }

    task->super.status = ucc_tl_ucp_test(task);
    if (task->super.status != UCC_OK) {
        return;
    }

    /* Need to check copy task if it is not in-place */
    if (task->allgather_linear.copy_task != NULL) {
        status = ctx->copy.test(ctx, task->allgather_linear.copy_task);
        if (status > 0) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        task->super.status = status;
        ctx->copy.finalize(task->allgather_linear.copy_task);
        task->allgather_linear.copy_task = NULL;
    }

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_linear_done", 0);
    return;
err:
    ucc_error("allgather linear progress failed with status %d: %s",
              task->super.status, ucc_status_string(task->super.status));
}
