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
 * This same machinery backs push-based alltoall (alltoall/alltoall_push.c):
 * counts and displacements are uniform, so the destination offset is
 * rank * chunk and the SETUP barrier is skipped (needs_setup == 0), giving
 * alltoall a single barrier.
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

/* Compute the chunk this rank pushes to `peer`: source bytes, source offset in
 * sbuf, and destination offset in the peer's rbuf. For alltoallv these come
 * from the variable counts/displacements (dst offset read from the peer's
 * rdispl_bytes exchanged in SETUP); for alltoall they are uniform. */
static inline void
alltoallv_push_peer_chunk(ucc_tl_cuda_task_t *task, ucc_rank_t peer,
                          size_t *send_bytes, size_t *src_displ,
                          size_t *dst_displ)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          rank     = UCC_TL_TEAM_RANK(team);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    size_t              sdt_size = ucc_dt_size(task->alltoallv_push.sdt);
    size_t              rdt_size = ucc_dt_size(task->alltoallv_push.rdt);

    if (task->alltoallv_push.needs_setup) {
        *send_bytes = sdt_size * (size_t)ucc_coll_args_get_count(
            args, task->alltoallv_push.scnts, peer);
        *src_displ = sdt_size * (size_t)ucc_coll_args_get_displacement(
            args, task->alltoallv_push.sdispl, peer);
        if (peer == rank) {
            *dst_displ = rdt_size * (size_t)ucc_coll_args_get_displacement(
                args, task->alltoallv_push.rdispl, rank);
        } else {
            *dst_displ = TASK_SYNC(task, peer)->alltoallv_ce.rdispl_bytes[rank];
        }
    } else {
        /* Uniform per-peer counts, but src and dst may use type-compatible
         * datatypes of different sizes: keep the source stride for send
         * offsets/bytes, and the destination stride for receive offsets. */
        size_t chunk     = sdt_size *
                           (args->src.info.count / UCC_TL_TEAM_SIZE(team));
        size_t dst_chunk = rdt_size *
                           (args->dst.info.count / UCC_TL_TEAM_SIZE(team));
        *send_bytes = chunk;
        *src_displ  = (size_t)peer * chunk;
        *dst_displ  = (size_t)rank * dst_chunk;
    }
}

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
                                        task->alltoallv_push.peer_map_raw[i],
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
    size_t                       rdt_size = ucc_dt_size(task->alltoallv_push.rdt);
    ucc_status_t                 status;
    ucc_rank_t                   peer;
    size_t                       send_bytes, src_displ, dst_displ;
    void                        *src, *dst;
    cudaError_t                  cuda_st;
    ucc_tl_cuda_sync_t          *sync;
    ucc_status_t                 local_err = UCC_OK;
    int                          err_slot;

    switch (task->alltoallv_push.stage) {
    case ALLTOALLV_PUSH_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        if (task->alltoallv_push.needs_setup) {
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
        } else {
            task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_PUSH;
        }
        /* fall through */
    case ALLTOALLV_PUSH_STAGE_SETUP:
        if (task->alltoallv_push.needs_setup) {
            status = ucc_tl_cuda_shm_barrier_test(rank, task->bar);
            if (status != UCC_OK) {
                task->super.status = status;
                return;
            }
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
                alltoallv_push_peer_chunk(task, peer, &send_bytes, &src_displ,
                                          &dst_displ);
                if (send_bytes == 0) {
                    continue;
                }
                src = PTR_OFFSET(task->alltoallv_push.sbuf, src_displ);
                dst = (peer == rank)
                      ? PTR_OFFSET(task->alltoallv_push.rbuf, dst_displ)
                      : PTR_OFFSET(task->alltoallv_push.peer_map_addr[peer],
                                   dst_displ);
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
                status = CUDA_FUNC(cudaMemcpyBatchAsync(
                    dsts, (const void *const *)srcs, sizes, n, attrs,
                    attr_idxs, n, stream));
                if (ucc_unlikely(status != UCC_OK)) {
                    local_err = status;
                    goto do_barrier;
                }
            }
        }
#else
        for (peer = 0; peer < nranks; peer++) {
            if (peer != rank &&
                !ucc_tl_cuda_team_topo_is_direct(
                    &team->super, team->topo, rank, peer)) {
                continue;
            }
            alltoallv_push_peer_chunk(task, peer, &send_bytes, &src_displ,
                                      &dst_displ);
            if (send_bytes == 0) {
                continue;
            }
            src = PTR_OFFSET(task->alltoallv_push.sbuf, src_displ);
            dst = (peer == rank)
                  ? PTR_OFFSET(task->alltoallv_push.rbuf, dst_displ)
                  : PTR_OFFSET(task->alltoallv_push.peer_map_addr[peer],
                               dst_displ);
            status = CUDA_FUNC(cudaMemcpyAsync(dst, src, send_bytes,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
            if (ucc_unlikely(status != UCC_OK)) {
                local_err = status;
                goto do_barrier;
            }
        }
#endif

        status = CUDA_FUNC(cudaEventRecord(evt, stream));
        if (ucc_unlikely(status != UCC_OK)) {
            local_err = status;
            goto do_barrier;
        }
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
            local_err = UCC_ERR_NO_MESSAGE;
        }
        /* fall through */
    do_barrier:
        err_slot = task->bar->local_sense[rank];
        if (local_err != UCC_OK) {
            /* Drain already-queued DMA before joining the barrier so the
               caller can safely release buffers/task after the error. */
            cudaError_t sync_st = cudaStreamSynchronize(stream);
            if (sync_st != cudaSuccess) {
                tl_error(UCC_TASK_LIB(task), "cudaStreamSynchronize failed: %s",
                         cudaGetErrorString(sync_st));
            }
            task->bar->error[err_slot] = local_err;
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
        err_slot = task->bar->local_sense[rank];
        break;
    }

    status = ucc_tl_cuda_shm_barrier_test(rank, task->bar);
    if (status == UCC_OK) {
        if (task->bar->error[err_slot] != UCC_OK) {
            status = task->bar->error[err_slot];
        }
        ucc_tl_cuda_put_sync(task);
        task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_SYNC;
    }
    task->super.status = status;
}

static ucc_status_t alltoallv_push_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    task->alltoallv_push.stage = ALLTOALLV_PUSH_STAGE_SYNC;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_alltoallv_push_setup(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t    *team = TASK_TEAM(task);
    ucc_coll_args_t       *args = &TASK_ARGS(task);
    ucc_rank_t             rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             i;
    ucc_status_t           status;
    ucc_tl_cuda_mem_info_t peer_mi;
    ucc_tl_cuda_cache_t   *cache;
    void                  *mapped;

    if (!UCC_TL_CUDA_TEAM_LIB(team)->cfg.alltoall_use_copy_engine) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(args->mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH) ||
        !(args->flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL) ||
        args->dst_memh.global_memh == NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Push writes to peer_map_addr + dst_displ, so it assumes the collective
     * rbuf starts exactly at the registered memh address. A memh covering a
     * broader region (rbuf is a sub-buffer) has no exchanged rbuf offset;
     * reject it rather than silently write before the peer's rbuf. */
    {
        ucc_mem_map_memh_t *local_memh =
            (ucc_mem_map_memh_t *)args->dst_memh.global_memh[rank];
        if (local_memh == NULL ||
            local_memh->address != task->alltoallv_push.rbuf) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    /* Push writes directly to peer rbuf and has no proxy path. On a topology
     * where any peer is not directly reachable, that peer's chunk would be
     * silently skipped (both here and in progress) while the final barrier
     * still returns UCC_OK, leaving its receive block stale. Only support
     * fully-connected CUDA topologies. */
    if (!ucc_tl_cuda_team_topo_is_fully_connected(team->topo)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->alltoallv_push.stage          = ALLTOALLV_PUSH_STAGE_SYNC;
    task->alltoallv_push.evt_completion = NULL;

    for (i = 0; i < UCC_TL_CUDA_MAX_PEERS; i++) {
        task->alltoallv_push.peer_map_addr[i] = NULL;
        task->alltoallv_push.peer_map_raw[i]  = NULL;
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
            goto err_unmap;
        }
        status = ucc_tl_cuda_mem_info_from_global_memh(
            args->dst_memh.global_memh, i, &peer_mi);
        if (ucc_unlikely(status != UCC_OK)) {
            goto err_unmap;
        }
        status = ucc_tl_cuda_map_memhandle(
            peer_mi.ptr, peer_mi.length, peer_mi.handle, &mapped, cache);
        if (ucc_unlikely(status != UCC_OK)) {
            goto err_unmap;
        }
        task->alltoallv_push.peer_map_raw[i]  = mapped;
        task->alltoallv_push.peer_map_addr[i] = PTR_OFFSET(mapped, peer_mi.offset);
    }

    status = ucc_ec_create_event(&task->alltoallv_push.evt_completion,
                                 UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        goto err_unmap;
    }

    task->super.post     = alltoallv_push_start;
    task->super.progress = alltoallv_push_progress;
    task->super.finalize = alltoallv_push_finalize;
    task->bar            = TASK_BAR(task);
    return UCC_OK;

err_unmap:
    /* Setup failed after mapping some peers. Roll back the mappings so cache
     * refcounts stay balanced; finalize is not installed yet, so the caller's
     * ucc_tl_cuda_task_put will not unmap them. */
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
                args->dst_memh.global_memh, i, &peer_mi) == UCC_OK) {
            ucc_tl_cuda_unmap_memhandle((uintptr_t)peer_mi.ptr,
                                        task->alltoallv_push.peer_map_raw[i],
                                        cache, 0);
        }
    }
    return status;
}

ucc_status_t ucc_tl_cuda_alltoallv_push_init(ucc_base_coll_args_t *coll_args,
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

    if (!UCC_COLL_ARGS_CONTIG_BUFFER(args)) {
        tl_debug(UCC_TL_TEAM_LIB(team), "alltoallv push: non-contiguous buffer");
        status = UCC_ERR_NOT_SUPPORTED;
        goto err;
    }

    task->alltoallv_push.sbuf            = args->src.info_v.buffer;
    task->alltoallv_push.rbuf            = args->dst.info_v.buffer;
    task->alltoallv_push.sdt             = args->src.info_v.datatype;
    task->alltoallv_push.rdt             = args->dst.info_v.datatype;
    task->alltoallv_push.scnts           = args->src.info_v.counts;
    task->alltoallv_push.rcnts           = args->dst.info_v.counts;
    task->alltoallv_push.sdispl          = args->src.info_v.displacements;
    task->alltoallv_push.rdispl          = args->dst.info_v.displacements;
    task->alltoallv_push.needs_setup     = 1;
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
