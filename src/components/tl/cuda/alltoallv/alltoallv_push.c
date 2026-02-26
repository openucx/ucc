/**
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

/*
 * Push-based alltoallv with two CPU SHM barriers.
 *
 * Peer destination buffers are pre-mapped at init time using global_memh_dst,
 * eliminating per-call cudaIpcGetMemHandle and cudaIpcOpenMemHandle. A single
 * lightweight SETUP barrier exchanges only receive displacements (no IPC
 * handles). Each rank then pushes its send chunks directly to peer destination
 * buffers from its own sbuf. A final barrier ensures all peers have finished
 * writing before completion.
 *
 * Compared to the CE algorithm:
 *  - No per-call cudaIpcGetMemHandle/OpenMemHandle (handles pre-mapped at init)
 *  - No cudaStreamWaitEvent (pushing from own sbuf, always ready)
 *  - SETUP barrier only exchanges rdispl_bytes[] — much lighter than CE's
 *    full IPC handle + event exchange
 *
 * Requirements:
 *  - global_memh_dst: peer dst buffer handles pre-exchanged at team level
 *  - No proxies
 */

#include "alltoallv.h"
#include "components/ec/ucc_ec.h"
#include "components/ec/cuda/ec_cuda_resources.h"
#include "core/ucc_ee.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

enum {
    ALLTOALLV_PUSH_STAGE_SYNC,   /* wait for free sync slot */
    ALLTOALLV_PUSH_STAGE_SETUP,  /* SHM barrier to exchange rdispl_bytes */
    ALLTOALLV_PUSH_STAGE_PUSH,   /* post push copies to GPU stream */
    ALLTOALLV_PUSH_STAGE_COPY,   /* poll until own copies complete */
    ALLTOALLV_PUSH_STAGE_BAR,    /* final barrier — all ranks done pushing */
};

static ucc_status_t alltoallv_push_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t  *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t  *team = TASK_TEAM(task);
    ucc_rank_t           rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_cache_t *cache;
    ucc_tl_cuda_mem_info_t mi;
    ucc_rank_t           i;

    tl_trace(UCC_TASK_LIB(task), "finalizing alltoallv push task %p", task);

    if (task->alltoallv_push.evt_completion) {
        ucc_ec_destroy_event(task->alltoallv_push.evt_completion,
                             UCC_EE_CUDA_STREAM);
        task->alltoallv_push.evt_completion = NULL;
    }

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == rank || task->alltoallv_push.peer_map_addr[i] == NULL) {
            continue;
        }
        if (!ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo, rank, i)) {
            continue;
        }
        cache = ucc_tl_cuda_get_cache(team, i);
        if (cache &&
            ucc_tl_cuda_mem_info_from_global_memh(
                task->alltoallv_push.global_memh_dst, i, &mi) == UCC_OK) {
            ucc_tl_cuda_unmap_memhandle((uintptr_t)mi.ptr,
                                        task->alltoallv_push.peer_map_addr[i],
                                        cache, 0);
        }
    }

    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

static void alltoallv_push_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t          *task     = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t          *team     = TASK_TEAM(task);
    ucc_rank_t                   rank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t                   nranks   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h                     ee       = task->super.ee;
    cudaStream_t                 stream   = ee ? (cudaStream_t)ee->ee_context
                                               : team->stream;
    ucc_ec_cuda_event_t         *ec_evt   = (ucc_ec_cuda_event_t *)
                                            task->alltoallv_push.evt_completion;
    cudaEvent_t                  evt      = ec_evt->event;
    ucc_coll_args_t             *args     = &TASK_ARGS(task);
    size_t                       sdt_size = ucc_dt_size(task->alltoallv_push.sdt);
    size_t                       rdt_size = ucc_dt_size(task->alltoallv_push.rdt);
    ucc_status_t                 status;
    ucc_rank_t                   peer;
    size_t                       send_bytes, src_displ, dst_displ;
    void                        *src, *dst;
    volatile ucc_tl_cuda_sync_t *peer_sync;
    cudaError_t                  cuda_st;
    ucc_tl_cuda_sync_t          *sync;

    switch (task->alltoallv_push.stage) {
    case ALLTOALLV_PUSH_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        /* Write our receive displacements so peers know where to push data
         * into our rbuf.  No IPC handle exchange needed — dst handles are
         * pre-mapped in peer_map_addr. */
        sync = TASK_SYNC(task, rank);
        for (ucc_rank_t i = 0; i < nranks; i++) {
            sync->alltoallv_ce.rdispl_bytes[i] =
                rdt_size * (size_t)ucc_coll_args_get_displacement(
                    args, task->alltoallv_push.rdispl, i);
        }
        ucc_memory_cpu_store_fence();
        status = ucc_tl_cuda_shm_barrier_start(rank, task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_SETUP;
        /* fall through */
    case ALLTOALLV_PUSH_STAGE_SETUP:
        status = ucc_tl_cuda_shm_barrier_test(rank, task->bar);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_PUSH;
        /* fall through */
    case ALLTOALLV_PUSH_STAGE_PUSH:
#if CUDART_VERSION >= 13000
        {
            const void                 *srcs[UCC_TL_CUDA_MAX_PEERS];
            void                       *dsts[UCC_TL_CUDA_MAX_PEERS];
            size_t                      sizes[UCC_TL_CUDA_MAX_PEERS];
            struct cudaMemcpyAttributes attrs[UCC_TL_CUDA_MAX_PEERS];
            size_t                      attr_idxs[UCC_TL_CUDA_MAX_PEERS];
            size_t                      n = 0;
            ucc_rank_t                  i;

            for (i = 0; i < nranks; i++) {
                peer = (rank + i) % nranks;
                if (peer != rank &&
                    !ucc_tl_cuda_team_topo_is_direct(
                        &team->super, team->topo, rank, peer)) {
                    continue;
                }
                send_bytes = sdt_size * (size_t)ucc_coll_args_get_count(
                    args, task->alltoallv_push.scnts, peer);
                if (send_bytes == 0) {
                    continue;
                }
                src_displ = sdt_size * (size_t)ucc_coll_args_get_displacement(
                    args, task->alltoallv_push.sdispl, peer);
                src = PTR_OFFSET(task->alltoallv_push.sbuf, src_displ);
                if (peer == rank) {
                    dst_displ = rdt_size * (size_t)ucc_coll_args_get_displacement(
                        args, task->alltoallv_push.rdispl, rank);
                    dst = PTR_OFFSET(task->alltoallv_push.rbuf, dst_displ);
                } else {
                    peer_sync = TASK_SYNC(task, peer);
                    dst_displ = peer_sync->alltoallv_ce.rdispl_bytes[rank];
                    dst       = PTR_OFFSET(task->alltoallv_push.peer_map_addr[peer],
                                           dst_displ);
                }
                srcs[n]  = src;
                dsts[n]  = dst;
                sizes[n] = send_bytes;
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
        for (peer = 0; peer < nranks; peer++) {
            if (peer != rank &&
                !ucc_tl_cuda_team_topo_is_direct(
                    &team->super, team->topo, rank, peer)) {
                continue;
            }
            send_bytes = sdt_size * (size_t)ucc_coll_args_get_count(
                args, task->alltoallv_push.scnts, peer);
            if (send_bytes == 0) {
                continue;
            }
            src_displ = sdt_size * (size_t)ucc_coll_args_get_displacement(
                args, task->alltoallv_push.sdispl, peer);
            src = PTR_OFFSET(task->alltoallv_push.sbuf, src_displ);
            if (peer == rank) {
                dst_displ = rdt_size * (size_t)ucc_coll_args_get_displacement(
                    args, task->alltoallv_push.rdispl, rank);
                dst = PTR_OFFSET(task->alltoallv_push.rbuf, dst_displ);
            } else {
                peer_sync = TASK_SYNC(task, peer);
                dst_displ = peer_sync->alltoallv_ce.rdispl_bytes[rank];
                dst       = PTR_OFFSET(task->alltoallv_push.peer_map_addr[peer],
                                       dst_displ);
            }
            CUDA_CHECK_GOTO(
                cudaMemcpyAsync(dst, src, send_bytes,
                                cudaMemcpyDeviceToDevice, stream),
                err, status);
        }
#endif

        CUDA_CHECK_GOTO(cudaEventRecord(evt, stream), err, status);
        task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_COPY;
        /* fall through */
    case ALLTOALLV_PUSH_STAGE_COPY:
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
        status = ucc_tl_cuda_shm_barrier_start(rank, task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_BAR;
        /* fall through */
    default:
        ucc_assert(task->alltoallv_push.stage == ALLTOALLV_PUSH_STAGE_BAR);
        break;
    }

    status = ucc_tl_cuda_shm_barrier_test(rank, task->bar);
    if (status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
        task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_SYNC;
    }
    task->super.status = status;
    return;
err:
    task->super.status = status;
}

static ucc_status_t alltoallv_push_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_SYNC;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_alltoallv_push_init(ucc_base_coll_args_t *coll_args,
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

    if (UCC_IS_INPLACE(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

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

    if (!UCC_COLL_ARGS_CONTIG_BUFFER(args)) {
        tl_debug(UCC_TL_TEAM_LIB(team), "alltoallv push: non-contiguous buffer");
        status = UCC_ERR_NOT_SUPPORTED;
        goto err;
    }

    task->alltoallv_push.stage           = ALLTOALLV_PUSH_STAGE_SYNC;
    task->alltoallv_push.sbuf            = args->src.info_v.buffer;
    task->alltoallv_push.rbuf            = args->dst.info_v.buffer;
    task->alltoallv_push.sdt             = args->src.info_v.datatype;
    task->alltoallv_push.rdt             = args->dst.info_v.datatype;
    task->alltoallv_push.scnts           = args->src.info_v.counts;
    task->alltoallv_push.rcnts           = args->dst.info_v.counts;
    task->alltoallv_push.sdispl          = args->src.info_v.displacements;
    task->alltoallv_push.rdispl          = args->dst.info_v.displacements;
    task->alltoallv_push.global_memh_dst = args->dst_memh.global_memh;
    task->alltoallv_push.evt_completion  = NULL;

    for (i = 0; i < UCC_TL_CUDA_MAX_PEERS; i++) {
        task->alltoallv_push.peer_map_addr[i] = NULL;
    }

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
        task->alltoallv_push.peer_map_addr[i] = PTR_OFFSET(mapped, peer_mi.offset);
    }

    status = ucc_ec_create_event(&task->alltoallv_push.evt_completion,
                                 UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        goto err;
    }

    task->super.post     = alltoallv_push_start;
    task->super.progress = alltoallv_push_progress;
    task->super.finalize = alltoallv_push_finalize;
    task->bar            = TASK_BAR(task);
    *task_p = &task->super;
    return UCC_OK;
err:
    ucc_tl_cuda_task_put(task);
    return status;
}
