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

ucc_status_t ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *ctask);

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

    task->barrier.phase = 0;

    task->super.super.status = UCC_INPROGRESS;
    status = ucc_tl_ucp_alltoall_onesided_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    task->super.super.status = status;
    ucc_task_complete(ctask);

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task   = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team   = TASK_TEAM(task);
    ptrdiff_t          src    = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest   = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_rank_t         mype   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         npes   = UCC_TL_TEAM_SIZE(team);
    size_t             nelems = TASK_ARGS(task).src.info.count;
    int *              phase  = &task->barrier.phase;
    long *             pSync;
    ucc_rank_t         peer;

    /* TODO: change when support for library-based work buffers is complete */
    pSync = TASK_ARGS(task).global_work_buffer;

    if (*phase == 1) {
        goto wait;
    }

    nelems = (nelems / npes) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    dest   = dest + mype * nelems;
    ucc_rank_t start = (mype + 1) % npes;
    ucc_tl_ucp_put_nb((void *)(src + start * nelems), (void *)dest, nelems,
                      start, team, task);
    ucc_tl_ucp_atomic_inc(pSync, start, team, task);

    for (peer = (start + 1) % npes; peer != start; peer = (peer + 1) % npes) {
        ucc_tl_ucp_put_nb((void *)(src + peer * nelems), (void *)dest, nelems,
                          peer, team, task);
        ucc_tl_ucp_atomic_inc(pSync, peer, team, task);
    }
    ucc_tl_ucp_flush(team);

    *phase = 1;

wait:
    if (*pSync < npes - 1) {
        return UCC_INPROGRESS;
    }

    *pSync                   = -1;
    task->super.super.status = UCC_OK;
    ucc_task_complete(ctask);
    return UCC_OK;
}
