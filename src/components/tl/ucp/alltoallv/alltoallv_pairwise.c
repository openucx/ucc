/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void ucc_tl_ucp_alltoallv_pairwise_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ptrdiff_t          sbuf  = (ptrdiff_t)TASK_ARGS(task).src.info_v.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)TASK_ARGS(task).dst.info_v.buffer;
    ucc_memory_type_t  smem  = TASK_ARGS(task).src.info_v.mem_type;
    ucc_memory_type_t  rmem  = TASK_ARGS(task).dst.info_v.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    int                polls = 0;
    ucc_rank_t         peer;
    int                posts, nreqs;
    size_t             rdt_size, sdt_size, data_size, data_displ;

    posts    = UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoallv_pairwise_num_posts;
    nreqs    = (posts > gsize || posts == 0) ? gsize : posts;
    rdt_size = ucc_dt_size(TASK_ARGS(task).dst.info_v.datatype);
    sdt_size = ucc_dt_size(TASK_ARGS(task).src.info_v.datatype);
    while ((task->tagged.send_posted < gsize ||
            task->tagged.recv_posted < gsize) &&
           (polls++ < task->n_polls)) {
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(team)->ucp_worker);
        while ((task->tagged.recv_posted < gsize) &&
               ((task->tagged.recv_posted - task->tagged.recv_completed) <
                nreqs)) {
            peer = get_recv_peer(grank, gsize, task->tagged.recv_posted);
            data_size =
                ucc_coll_args_get_count(
                    &TASK_ARGS(task), TASK_ARGS(task).dst.info_v.counts, peer) *
                rdt_size;
            data_displ = ucc_coll_args_get_displacement(
                             &TASK_ARGS(task),
                             TASK_ARGS(task).dst.info_v.displacements, peer) *
                         rdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nz((void *)(rbuf + data_displ),
                                             data_size, rmem, peer, team, task),
                          task, out);
            polls = 0;
        }
        while ((task->tagged.send_posted < gsize) &&
               ((task->tagged.send_posted - task->tagged.send_completed) <
                nreqs)) {
            peer = get_send_peer(grank, gsize, task->tagged.send_posted);
            data_size =
                ucc_coll_args_get_count(
                    &TASK_ARGS(task), TASK_ARGS(task).src.info_v.counts, peer) *
                sdt_size;
            data_displ = ucc_coll_args_get_displacement(
                             &TASK_ARGS(task),
                             TASK_ARGS(task).src.info_v.displacements, peer) *
                         sdt_size;
            UCPCHECK_GOTO(ucc_tl_ucp_send_nz((void *)(sbuf + data_displ),
                                             data_size, smem, peer, team, task),
                          task, out);
            polls = 0;
        }
    }
    if ((task->tagged.send_posted < gsize) ||
        (task->tagged.recv_posted < gsize)) {
        return;
    }
    task->super.status = ucc_tl_ucp_test(task);
out:
    if (task->super.status != UCC_INPROGRESS) {
        UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task,
                                         "ucp_alltoallv_pairwise_done", 0);
    }
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_alltoallv_pairwise_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t   *args = &TASK_ARGS(task);

    task->super.post     = ucc_tl_ucp_alltoallv_pairwise_start;
    task->super.progress = ucc_tl_ucp_alltoallv_pairwise_progress;

    task->n_polls = ucc_min(1, task->n_polls);
    if (UCC_TL_UCP_TEAM_CTX(team)->cfg.pre_reg_mem) {
        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->src.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                               size) *
                 ucc_dt_size(args->src.info_v.datatype)),
                args->src.info_v.mem_type);
        }

        if (args->flags & UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER) {
            ucc_tl_ucp_pre_register_mem(
                team, args->dst.info_v.buffer,
                (ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                               size) *
                 ucc_dt_size(args->dst.info_v.datatype)),
                args->dst.info_v.mem_type);
        }
    }
    return UCC_OK;
}
