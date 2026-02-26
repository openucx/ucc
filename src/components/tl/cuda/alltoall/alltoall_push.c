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
 * Requirements:
 *  - global_memh_dst: peer destination buffer handles pre-exchanged.
 *  - No proxies (push writes directly to peer rbuf, not via a proxy rank).
 */

#include "alltoall.h"
#include "components/ec/ucc_ec.h"
#include "components/ec/cuda/ec_cuda_resources.h"
#include "core/ucc_ee.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cuda_def.h"

enum {
    ALLTOALL_PUSH_STAGE_SYNC, /*< wait for free sync slot */
    ALLTOALL_PUSH_STAGE_PUSH, /*< post push copies to GPU stream, record event */
    ALLTOALL_PUSH_STAGE_COPY, /*< poll event until own copies complete */
    ALLTOALL_PUSH_STAGE_BAR,  /*< one SHM barrier — all ranks done pushing */
};

static ucc_status_t alltoall_push_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t    *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t    *team = TASK_TEAM(task);
    ucc_rank_t             rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_cache_t   *cache;
    ucc_tl_cuda_mem_info_t mi;
    ucc_rank_t             i;

    tl_trace(UCC_TASK_LIB(task), "finalizing push task %p", task);

    if (task->alltoall_push.evt_completion) {
        ucc_ec_destroy_event(task->alltoall_push.evt_completion,
                             UCC_EE_CUDA_STREAM);
        task->alltoall_push.evt_completion = NULL;
    }

    /* Release pre-mapped peer dst handles. */
    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == rank || task->alltoall_push.peer_map_addr[i] == NULL) {
            continue;
        }
        if (!ucc_tl_cuda_team_topo_is_direct(
                &team->super, team->topo, rank, i)) {
            continue;
        }
        cache = ucc_tl_cuda_get_cache(team, i);
        if (cache) {
            if (ucc_tl_cuda_mem_info_from_global_memh(
                    task->alltoall_push.global_memh_dst, i, &mi) == UCC_OK) {
                ucc_tl_cuda_unmap_memhandle(
                    (uintptr_t)mi.ptr,
                    task->alltoall_push.peer_map_addr[i],
                    cache,
                    0);
            }
        }
    }

    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

static void alltoall_push_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t  *task   = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t  *team   = TASK_TEAM(task);
    ucc_rank_t           rank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t           nranks = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h             ee     = task->super.ee;
    cudaStream_t         stream = ee ? (cudaStream_t)ee->ee_context : team->stream;
    cudaEvent_t          evt;
    ucc_ec_cuda_event_t *ec_evt;
    size_t               chunk_size;
    ucc_status_t         status;
    ucc_rank_t           peer;
    size_t               src_off;
    size_t               dst_off;
    void                *src;
    void                *dst;
    cudaError_t          cuda_st;

    ec_evt     = (ucc_ec_cuda_event_t *) task->alltoall_push.evt_completion;
    evt        = ec_evt->event;
    chunk_size = ucc_dt_size(task->alltoall_push.dt) *
                 (TASK_ARGS(task).src.info.count / nranks);

    switch (task->alltoall_push.stage) {
    case ALLTOALL_PUSH_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        task->alltoall_push.stage = ALLTOALL_PUSH_STAGE_PUSH;
        /* fall through */
    case ALLTOALL_PUSH_STAGE_PUSH:
#if CUDART_VERSION >= 13000
        /* Batch all push copies so the runtime can execute them in parallel
         * across independent NVLink connections (same as CE's batch path). */
        {
            const void                 *srcs[UCC_TL_CUDA_MAX_PEERS];
            void                       *dsts[UCC_TL_CUDA_MAX_PEERS];
            size_t                      sizes[UCC_TL_CUDA_MAX_PEERS];
            struct cudaMemcpyAttributes attrs[UCC_TL_CUDA_MAX_PEERS];
            size_t                      attr_idxs[UCC_TL_CUDA_MAX_PEERS];
            size_t                      n = 0;
            ucc_rank_t                  i = 0;

            for (i = 0; i < nranks; i++) {
                peer = (rank + i) % UCC_TL_TEAM_SIZE(team);

                if (!ucc_tl_cuda_team_topo_is_direct(
                        &team->super, team->topo, rank, peer) || chunk_size == 0)
                    continue;
                src_off  = (size_t)peer * chunk_size;
                dst_off  = (size_t)rank * chunk_size;
                src      = PTR_OFFSET(task->alltoall_push.sbuf, src_off);
                dst      = (peer == rank)
                           ? PTR_OFFSET(task->alltoall_push.rbuf,       dst_off)
                           : PTR_OFFSET(task->alltoall_push.peer_map_addr[peer], dst_off);
                srcs[n]  = src;
                dsts[n]  = dst;
                sizes[n] = chunk_size;
                memset(&attrs[n], 0, sizeof(attrs[n]));
                attrs[n].srcAccessOrder = cudaMemcpySrcAccessOrderAny;
                attrs[n].flags          = cudaMemcpyFlagPreferOverlapWithCompute;
                attr_idxs[n]            = n;
                n++;
            }
            if (n > 0) {
                CUDA_CHECK_GOTO(
                    cudaMemcpyBatchAsync(dsts, (const void *const *)srcs,
                                        sizes, n, attrs, attr_idxs, n, stream),
                    err, status);
            }
        }
#else
        /* Fallback: sequential copies (no cudaMemcpyBatchAsync). */
        for (peer = 0; peer < nranks; peer++) {
            if (!ucc_tl_cuda_team_topo_is_direct(
                    &team->super, team->topo, rank, peer) || chunk_size == 0)
                continue;
            src_off = (size_t)peer * chunk_size;
            dst_off = (size_t)rank * chunk_size;
            src     = PTR_OFFSET(task->alltoall_push.sbuf, src_off);
            dst     = (peer == rank)
                      ? PTR_OFFSET(task->alltoall_push.rbuf,       dst_off)
                      : PTR_OFFSET(task->alltoall_push.peer_map_addr[peer], dst_off);
            CUDA_CHECK_GOTO(
                cudaMemcpyAsync(dst, src, chunk_size,
                                cudaMemcpyDeviceToDevice, stream),
                err, status);
        }
#endif

        CUDA_CHECK_GOTO(cudaEventRecord(evt, stream), err, status);
        task->alltoall_push.stage = ALLTOALL_PUSH_STAGE_COPY;
        /* fall through */
    case ALLTOALL_PUSH_STAGE_COPY:
        cuda_st = cudaEventQuery(evt);
        if (cuda_st == cudaErrorNotReady) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        if (cuda_st != cudaSuccess) {
            tl_error(UCC_TASK_LIB(task), "cudaEventQuery failed: %s",
                     cudaGetErrorString(cuda_st));
            task->super.status = UCC_ERR_NO_MESSAGE;
            return;
        }
        status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoall_push.stage = ALLTOALL_PUSH_STAGE_BAR;
        /* fall through */
    default:
        ucc_assert(task->alltoall_push.stage == ALLTOALL_PUSH_STAGE_BAR);
        break;
    }

    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
        task->alltoall_push.stage = ALLTOALL_PUSH_STAGE_SYNC;
    }
    task->super.status = status;
    return;
err:
    task->super.status = status;
}

static ucc_status_t alltoall_push_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    task->alltoall_push.stage = ALLTOALL_PUSH_STAGE_SYNC;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_alltoall_push_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *tl_team,
                                             ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t    *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t    *task;
    ucc_coll_args_t       *args;
    ucc_rank_t             rank;
    ucc_rank_t             i;
    ucc_status_t           status;
    ucc_tl_cuda_mem_info_t peer_mi;
    ucc_tl_cuda_cache_t   *cache;
    void                  *mapped;

    if (UCC_IS_INPLACE(coll_args->args))
        return UCC_ERR_NOT_SUPPORTED;

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK))
        return status;

    args = &TASK_ARGS(task);
    rank = UCC_TL_TEAM_RANK(team);

    if (!UCC_TL_CUDA_TEAM_LIB(team)->cfg.alltoall_use_copy_engine) {
        status = UCC_ERR_NOT_SUPPORTED;
        goto err;
    }

    if (!(args->mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH) ||
        !(args->flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL) ||
        args->dst_memh.global_memh == NULL) {
        status = UCC_ERR_NOT_SUPPORTED;
        goto err;
    }

    task->alltoall_push.stage           = ALLTOALL_PUSH_STAGE_SYNC;
    task->alltoall_push.sbuf            = args->src.info.buffer;
    task->alltoall_push.rbuf            = args->dst.info.buffer;
    task->alltoall_push.dt              = args->src.info.datatype;
    task->alltoall_push.global_memh_dst = args->dst_memh.global_memh;
    task->alltoall_push.evt_completion  = NULL;

    for (i = 0; i < UCC_TL_CUDA_MAX_PEERS; i++)
        task->alltoall_push.peer_map_addr[i] = NULL;

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == rank) {
            continue;
        }
        if (!ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo, rank, i)) {
            continue;
        }
        cache = ucc_tl_cuda_get_cache(team, i);
        if (ucc_unlikely(!cache)) {
            status = UCC_ERR_NO_MESSAGE;
            goto err;
        }
        status = ucc_tl_cuda_mem_info_from_global_memh(
            args->dst_memh.global_memh, i, &peer_mi);
        if (ucc_unlikely(status != UCC_OK)) {
            goto err;
        }
        status = ucc_tl_cuda_map_memhandle(
            peer_mi.ptr, peer_mi.length, peer_mi.handle, &mapped, cache);
        if (ucc_unlikely(status != UCC_OK)) {
            goto err;
        }
        task->alltoall_push.peer_map_addr[i] = PTR_OFFSET(mapped, peer_mi.offset);
    }

    status = ucc_ec_create_event(&task->alltoall_push.evt_completion,
                                 UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        goto err;
    }

    task->super.post     = alltoall_push_start;
    task->super.progress = alltoall_push_progress;
    task->super.finalize = alltoall_push_finalize;
    task->bar            = TASK_BAR(task);
    *task_p = &task->super;
    return UCC_OK;
err:
    ucc_tl_cuda_task_put(task);
    return status;
}
