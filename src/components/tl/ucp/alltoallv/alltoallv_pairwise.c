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
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"

static inline ucc_rank_t get_recv_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank + step) % size;
}

static inline ucc_rank_t get_send_peer(ucc_rank_t rank, ucc_rank_t size,
                                       ucc_rank_t step)
{
    return (rank - step + size) % size;
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = task->team;
    ptrdiff_t          sbuf  = (ptrdiff_t)task->args.src.info_v.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)task->args.dst.info_v.buffer;
    ucc_rank_t         grank = team->rank;
    ucc_rank_t         gsize = team->size;
    int                polls = 0;
    ucc_rank_t         peer;
    int                posts, nreqs;//, count_stride, displ_stride;
    size_t             rdt_size, sdt_size, data_size, data_displ;

    posts    = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    nreqs    = (posts > gsize || posts == 0) ? gsize : posts;
    rdt_size = ucc_dt_size(task->args.src.info_v.datatype);
    sdt_size = ucc_dt_size(task->args.dst.info_v.datatype);
    while ((task->send_posted < gsize || task->recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->recv_posted < gsize) &&
               ((task->recv_posted - task->recv_completed) < nreqs)) {
            peer       = get_recv_peer(grank, gsize, task->recv_posted);
            data_size  = ucc_coll_args_get_count(&task->args,
                            task->args.dst.info_v.counts, peer) * rdt_size;
            data_displ = ucc_coll_args_get_displacement(&task->args,
                            task->args.dst.info_v.displacements,
                            peer) * rdt_size;
            ucc_tl_ucp_recv_nb((void *)(rbuf + data_displ), data_size,
                               task->args.dst.info_v.mem_type, peer, team,
                               task);
            polls = 0;
        }
        while ((task->send_posted < gsize) &&
               ((task->send_posted - task->send_completed) < nreqs)) {
            peer       = get_send_peer(grank, gsize, task->send_posted);
            data_size  = ucc_coll_args_get_count(&task->args,
                            task->args.src.info_v.counts, peer) * sdt_size;
            data_displ = ucc_coll_args_get_displacement(&task->args,
                            task->args.src.info_v.displacements,
                            peer) * sdt_size;
            ucc_tl_ucp_send_nb((void *)(sbuf + data_displ), data_size,
                               task->args.src.info_v.mem_type, peer, team,
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
    task->n_polls            = ucc_min(1, task->n_polls);

    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        if (task->args.flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER) {
            ucc_tl_ucp_pre_register_mem(team, task->args.src.info_v.buffer,
                                        (ucc_coll_args_get_total_count(&task->args,
                                         task->args.src.info_v.counts, team->size ) *
                                         ucc_dt_size(task->args.src.info_v.datatype)),
                                        task->args.src.info_v.mem_type);
        }

        if (task->args.flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER) {
            ucc_tl_ucp_pre_register_mem(team, task->args.dst.info_v.buffer,
                                        (ucc_coll_args_get_total_count(&task->args,
                                         task->args.dst.info_v.counts, team->size) *
                                         ucc_dt_size(task->args.dst.info_v.datatype)),
                                        task->args.dst.info_v.mem_type);
        }
    }

    ucc_tl_ucp_alltoallv_pairwise_progress(&task->super);
    if (UCC_INPROGRESS == task->super.super.status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (task->super.super.status < 0) {
        return task->super.super.status;
    }
    return UCC_OK;
}
