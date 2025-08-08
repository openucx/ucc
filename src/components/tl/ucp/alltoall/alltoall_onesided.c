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
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"

void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *ctask);

static ucc_status_t ucc_tl_ucp_alltoall_onesided_start_ops(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t  *task     = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ucc_status_t        status;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    if (task->flags & UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG) {
        status = ucc_tl_ucp_coll_dynamic_segment_exchange_nb(task);
        if (status == UCC_INPROGRESS) {
            return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        }
        if (UCC_OK != status) {
            task->super.status = status;
            return task->super.status;
        }
    }

    /* If using dynamic segments, wait for exchange to complete before starting operations */
    if (task->flags & UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG) {
        return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    }

    /* Start the onesided operations */
    return ucc_tl_ucp_alltoall_onesided_start_ops(task);
}

static ucc_status_t ucc_tl_ucp_alltoall_onesided_start_ops(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t  *team     = TASK_TEAM(task);
    ptrdiff_t           src      = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t           dest     = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    size_t              nelems   = TASK_ARGS(task).src.info.count;
    ucc_rank_t          grank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          gsize    = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          start    = (grank + 1) % gsize;
    long               *pSync    = TASK_ARGS(task).global_work_buffer;
    ucc_mem_map_mem_h   src_memh = TASK_ARGS(task).src_memh.local_memh;
    ucc_mem_map_mem_h  *dst_memh = TASK_ARGS(task).dst_memh.global_memh;
    ucc_rank_t          peer;


    if (task->flags & UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG) {
        src_memh = task->dynamic_segments.src_global[grank];
        dst_memh = (ucc_mem_map_mem_h *)task->dynamic_segments.dst_global;
    } else {
        if (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL) {
            src_memh = TASK_ARGS(task).src_memh.global_memh[grank];
        }
    }

    /* TODO: change when support for library-based work buffers is complete */
    nelems = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    dest   = dest + grank * nelems;
    for (peer = start; task->onesided.put_posted < gsize;
         peer = (peer + 1) % gsize) {
        UCPCHECK_GOTO(ucc_tl_ucp_put_nb((void *)(src + peer * nelems),
                                        (void *)dest, nelems, peer, src_memh,
                                        dst_memh, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_atomic_inc(pSync, peer, dst_memh, team), task,
                      out);
    }

    /* Operations posted, return UCC_INPROGRESS to let progress function set flag */
    return UCC_INPROGRESS;
out:
    return task->super.status;
}

void ucc_tl_ucp_alltoall_onesided_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    long              *pSync = TASK_ARGS(task).global_work_buffer;

    /* Handle dynamic segment exchange if needed */
    if (task->flags & UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG) {
        ucc_status_t status = ucc_tl_ucp_test_dynamic_segment(task);
        if (status == UCC_INPROGRESS) {
            return;
        }
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        /* Dynamic segment exchange is complete, start the onesided operations */
        if (!(task->flags & UCC_TL_UCP_TASK_FLAG_OPS_STARTED)) {
            ucc_tl_ucp_alltoall_onesided_start_ops(task);
            task->flags |= UCC_TL_UCP_TASK_FLAG_OPS_STARTED;
            return; /* Return to wait for operations to complete */
        }
    }
    ucc_status_t test_status = ucc_tl_ucp_test_onesided(task, gsize);
    if (test_status == UCC_INPROGRESS) {
        return;
    }
    pSync[0]           = 0;
    task->super.status = ucc_tl_ucp_coll_dynamic_segment_finalize(task);
}
