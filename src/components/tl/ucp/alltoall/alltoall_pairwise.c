/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "tl_ucp_sendrecv.h"

static inline int get_recv_peer(int rank, int size, int step)
{
    return (rank + step) % size;
}

static inline int get_send_peer(int rank, int size, int step)
{
    return (rank - step + size) % size;
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = task->team;
    ptrdiff_t          sbuf  = (ptrdiff_t)task->args.src.info.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)task->args.dst.info.buffer;
    int                grank = team->rank;
    int                gsize = team->size;
    int                polls = 0;
    int                peer, posts, nreqs;
    size_t             data_size;

    posts     = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_pairwise_num_posts;
    nreqs     = (posts > gsize || posts == 0) ? gsize : posts;
    data_size = (size_t)task->args.src.info.count *
                ucc_dt_size(task->args.src.info.datatype);
    while ((task->send_posted < gsize || task->recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->recv_posted < gsize) &&
               ((task->recv_posted - task->recv_completed) < nreqs)) {
            peer = get_recv_peer(grank, gsize, task->recv_posted);
            ucc_tl_ucp_recv_nb((void *)(rbuf + peer * data_size), data_size,
                               task->args.dst.info.mem_type, peer, team, task);
            polls = 0;
        }
        while ((task->send_posted < gsize) &&
               ((task->send_posted - task->send_completed) < nreqs)) {
            peer = get_send_peer(grank, gsize, task->send_posted);
            ucc_tl_ucp_send_nb((void *)(sbuf + peer * data_size), data_size,
                               task->args.src.info.mem_type, peer, team, task);
            polls = 0;
        }
    }
    if ((task->send_posted < gsize) || (task->recv_posted < gsize)) {
        return task->super.super.status;
    }

    task->super.super.status = ucc_tl_ucp_test(task);
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_alltoall_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;

    task->super.super.status = UCC_INPROGRESS;
    task->n_polls            = ucc_min(1, task->n_polls);
    ucc_tl_ucp_alltoall_pairwise_progress(&task->super);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    }
    else if (task->super.super.status < 0) {
        return task->super.super.status;
    }
    return UCC_OK;
}
