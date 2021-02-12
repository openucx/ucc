/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
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

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t  *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t  *team  = task->team;
    ptrdiff_t           sbuf  = (ptrdiff_t)task->args.buffer_info.src_buffer;
    ptrdiff_t           rbuf  = (ptrdiff_t)task->args.buffer_info.dst_buffer;
    int                 grank = team->rank;
    int                 gsize = team->size;
    int                 polls = 0;
    int peer, chunk, nreqs;
    size_t rdt_size, sdt_size, data_size;
    ucc_aint_t *src_displ, *rcv_displ;

    chunk = UCC_TL_UCP_TEAM_CTX(team)->cfg.alltoallv_pairwise_chunk;
    nreqs = (chunk > gsize || chunk == 0) ? gsize: chunk;
    rdt_size = ucc_dt_size(task->args.buffer_info.dst_datatype);
    sdt_size = ucc_dt_size(task->args.buffer_info.src_datatype);
    src_displ = task->args.buffer_info.src_displacements;
    rcv_displ = task->args.buffer_info.dst_displacements;
    while ((task->send_posted < gsize || task->recv_posted < gsize) &&
           (polls++ < task->n_polls+1)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->recv_posted < gsize) &&
              ((task->recv_posted - task->recv_completed) < nreqs)) {
            peer = get_recv_peer(grank, gsize, task->recv_posted);
            data_size = task->args.buffer_info.dst_counts[peer] * rdt_size;
            ucc_tl_ucp_recv_nb((void*)(rbuf + rcv_displ[peer] * rdt_size),
                               data_size, UCC_MEMORY_TYPE_UNKNOWN, peer, team,
                               task);
            polls = 0;
        }
        while ((task->send_posted < gsize) &&
              ((task->send_posted - task->send_completed) < nreqs)) {
            peer = get_send_peer(grank, gsize, task->send_posted);
            data_size = task->args.buffer_info.src_counts[peer] * sdt_size;
            ucc_tl_ucp_send_nb((void*)(sbuf + src_displ[peer] * sdt_size),
                               data_size, UCC_MEMORY_TYPE_UNKNOWN, peer, team,
                               task);
            polls = 0;
        }
    }
    if ((task->send_posted < gsize) || (task->recv_posted < gsize)) {
        return task->super.super.status;
    }

    task->super.super.status = ucc_tl_ucp_test(task);
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;

    task->super.super.status = UCC_INPROGRESS;
    ucc_tl_ucp_alltoallv_pairwise_progress(&task->super);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (task->super.super.status < 0) {
        return task->super.super.status;
    }
    return UCC_OK;
}
