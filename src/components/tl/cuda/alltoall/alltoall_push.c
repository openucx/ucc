/**
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

/*
 * Push-based alltoall with one CPU SHM barrier.
 *
 * Each rank pushes its chunks directly to peer destination buffers. Once all
 * pushes complete (detected via cudaEventQuery on the rank's own stream), the
 * rank enters a single CPU SHM barrier. When all ranks have arrived, every
 * rbuf is fully populated and UCC_OK can be returned.
 *
 * This replaces the pull-based CE algorithm's TWO barriers with ONE:
 *  - No SETUP barrier: we push from our own sbuf so no cudaStreamWaitEvent on
 *    peer events is needed, eliminating the stale-event problem entirely.
 *  - ONE FINAL barrier: ensures all peers have completed their pushes into our
 *    rbuf before we signal completion to the user.
 *
 * Counts and displacements are uniform, so the destination offset is
 * rank * chunk and no SETUP exchange is required (needs_setup == 0). The
 * shared stage machine lives in alltoallv/alltoallv_push.c; this file only
 * does the alltoall-specific argument parsing.
 *
 * Requirements:
 *  - global_memh_dst: peer destination buffer handles pre-exchanged.
 *  - No proxies (push writes directly to peer rbuf, not via a proxy rank).
 */

#include "alltoall.h"
#include "../alltoallv/alltoallv.h"

ucc_status_t ucc_tl_cuda_alltoall_push_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *tl_team,
                                             ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_coll_args_t    *args;
    ucc_status_t        status;

    if (UCC_IS_INPLACE(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    args = &TASK_ARGS(task);

    /* Uniform counts: chunk = count/nranks, dst offset = rank*chunk — both
     * computable locally, so the SETUP barrier is skipped. */
    task->alltoallv_push.sbuf            = args->src.info.buffer;
    task->alltoallv_push.rbuf            = args->dst.info.buffer;
    task->alltoallv_push.sdt             = args->src.info.datatype;
    task->alltoallv_push.rdt             = args->dst.info.datatype;
    task->alltoallv_push.scnts           = NULL;
    task->alltoallv_push.rcnts           = NULL;
    task->alltoallv_push.sdispl          = NULL;
    task->alltoallv_push.rdispl          = NULL;
    task->alltoallv_push.needs_setup     = 0;
    task->alltoallv_push.global_memh_dst = args->dst_memh.global_memh;

    status = ucc_tl_cuda_alltoallv_push_setup(task);
    if (ucc_unlikely(status != UCC_OK)) {
        goto err;
    }

    *task_p = &task->super;
    return UCC_OK;
err:
    ucc_tl_cuda_task_put(task);
    return status;
}
