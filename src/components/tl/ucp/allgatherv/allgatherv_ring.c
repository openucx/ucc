/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"

ucc_status_t ucc_tl_ucp_allgatherv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team     = task->team;
    ucc_rank_t         grank    = team->rank;
    ucc_rank_t         gsize    = team->size;
    ptrdiff_t          rbuf     = (ptrdiff_t)task->args.dst.info_v.buffer;
    ucc_memory_type_t  rmem     = task->args.dst.info_v.mem_type;
    size_t             rdt_size = ucc_dt_size(task->args.dst.info_v.datatype);
    ucc_rank_t         sendto   = (grank + 1) % gsize;
    ucc_rank_t         recvfrom = (grank - 1 + gsize) % gsize;
    ucc_rank_t         send_idx, recv_idx;
    size_t             data_size, data_displ;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return task->super.super.status;
    }
    while (task->send_posted < gsize) {
        send_idx = (grank - task->send_posted + 1 + gsize) % gsize;
        data_displ = ucc_coll_args_get_displacement(&task->args,
                        task->args.dst.info_v.displacements, send_idx) *
                        rdt_size;
        data_size = ucc_coll_args_get_count(&task->args,
                        task->args.dst.info_v.counts, send_idx) * rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(rbuf + data_displ), data_size,
                                         rmem, sendto, team, task),
                      task, out);
        recv_idx = (grank - task->recv_posted + gsize) % gsize;
        data_displ = ucc_coll_args_get_displacement(&task->args,
                        task->args.dst.info_v.displacements, recv_idx) *
                        rdt_size;
        data_size = ucc_coll_args_get_count(&task->args,
                        task->args.dst.info_v.counts, recv_idx) * rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + data_displ), data_size,
                                         rmem, recvfrom, team, task),
                      task, out);
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return task->super.super.status;
        }
    }
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.super.status = UCC_OK;
out:
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = task->team;
    ptrdiff_t          sbuf  = (ptrdiff_t)task->args.src.info.buffer;
    ptrdiff_t          rbuf  = (ptrdiff_t)task->args.dst.info_v.buffer;
    ucc_memory_type_t  smem  = task->args.src.info.mem_type;
    ucc_memory_type_t  rmem  = task->args.dst.info_v.mem_type;
    ucc_rank_t         grank = team->rank;
    size_t             data_size, data_displ, rdt_size;
    ucc_status_t       status;

    task->super.super.status     = UCC_INPROGRESS;
    if (!UCC_IS_INPLACE(task->args)) {
        /* TODO replace local sendrecv with memcpy? */
        rdt_size   = ucc_dt_size(task->args.dst.info_v.datatype);
        data_displ = ucc_coll_args_get_displacement(
                        &task->args, task->args.dst.info_v.displacements,
                        grank) * rdt_size;
        data_size  = ucc_coll_args_get_count(
                        &task->args, task->args.dst.info_v.counts,
                        grank) *rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)rbuf + data_displ, data_size,
                                         rmem, grank, team, task),
                      task, error);
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)sbuf, data_size, smem, grank,
                                         team, task),
                      task, error);
    } else {
        /* to simplify progress fucnction and make it identical for
           in-place and non in-place */
        task->send_posted = task->recv_posted = 1;
        task->send_completed = task->recv_completed = 1;
    }

    status = ucc_tl_ucp_allgatherv_ring_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
error:
    return task->super.super.status;
}
