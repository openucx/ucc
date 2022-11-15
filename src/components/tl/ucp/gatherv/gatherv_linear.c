/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "gatherv.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"

static inline ucc_rank_t get_peer(ucc_rank_t rank, ucc_rank_t size,
                                  ucc_rank_t step)
{
    return (rank + step) % size;
}

void ucc_tl_ucp_gatherv_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    void*              rbuf  = args->dst.info_v.buffer;
    ucc_memory_type_t  rmem  = args->dst.info_v.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    int                polls = 0;
    ucc_rank_t         peer, posts, nreqs;
    size_t             dt_size, data_size, data_displ;

    if (UCC_IS_ROOT(*args, grank)) {
        posts   = UCC_TL_UCP_TEAM_LIB(team)->cfg.gatherv_linear_num_posts;
        nreqs   = (posts > gsize || posts == 0) ? gsize : posts;
        dt_size = ucc_dt_size(TASK_ARGS(task).dst.info_v.datatype);

        while (polls++ < task->n_polls) {
            ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->worker.ucp_worker);
            while ((task->tagged.recv_posted < gsize) &&
                   ((task->tagged.recv_posted - task->tagged.recv_completed) <
                    nreqs)) {
                peer       = get_peer(grank, gsize, task->tagged.recv_posted);
                data_size  = ucc_coll_args_get_count(args,
                                args->dst.info_v.counts, peer) * dt_size;
                data_displ = ucc_coll_args_get_displacement(args,
                                args->dst.info_v.displacements, peer) * dt_size;
                UCPCHECK_GOTO(ucc_tl_ucp_recv_nz(PTR_OFFSET(rbuf, data_displ),
                                                 data_size, rmem, peer, team, task),
                               task, out);
                polls = 0;
            }
        }
        if (task->tagged.recv_posted < gsize) {
            return;
        }
    }

    task->super.status = ucc_tl_ucp_test(task);
out:
    if (task->super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_gatherv_linear_done", 0);
    }
}

ucc_status_t ucc_tl_ucp_gatherv_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_memory_type_t  smem  = args->src.info.mem_type;
    ucc_memory_type_t  rmem  = args->dst.info_v.mem_type;
    void *             sbuf  = args->src.info.buffer;
    void              *rbuf;
    size_t             dt_size, data_displ, data_size;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_gatherv_linear_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (UCC_IS_ROOT(*args, grank)) {
        if (!UCC_IS_INPLACE(*args)) {
            dt_size    = ucc_dt_size(args->dst.info_v.datatype);
            data_size  = ucc_coll_args_get_count(args,
                            args->dst.info_v.counts, grank) * dt_size;
            data_displ = ucc_coll_args_get_displacement(args,
                            args->dst.info_v.displacements, grank) * dt_size;
            rbuf       = PTR_OFFSET(args->dst.info_v.buffer, data_displ);

            UCPCHECK_GOTO(ucc_tl_ucp_recv_nz(rbuf, data_size, rmem, grank, team,
                                             task),
                          task, error);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nz(sbuf, data_size, smem, grank, team,
                                             task),
                          task, error);
        } else {
            /* to simplify progress fucnction and make it identical for
            in-place and non in-place */
            task->tagged.send_posted = task->tagged.send_completed = 1;
            task->tagged.recv_posted = task->tagged.recv_completed = 1;
        }
    } else {
            dt_size   = ucc_dt_size(args->src.info.datatype);
            data_size = args->src.info.count * dt_size;

            UCPCHECK_GOTO(ucc_tl_ucp_send_nz(sbuf, data_size, smem, args->root,
                                             team, task),
                          task, error);
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
error:
    return task->super.status;

}

ucc_status_t ucc_tl_ucp_gatherv_linear_init(ucc_tl_ucp_task_t *task)
{
    task->super.post     = ucc_tl_ucp_gatherv_linear_start;
    task->super.progress = ucc_tl_ucp_gatherv_linear_progress;

    task->n_polls = ucc_max(1, task->n_polls);
    return UCC_OK;
}
