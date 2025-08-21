/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "alltoallv.h"
#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"
#include "utils/ucc_compiler_def.h"

#include <cuda_runtime.h>
#include <driver_types.h>
#include <string.h>


enum {
    ALLTOALL_CE_STAGE_SYNC,  /*< Wait for free SYNC segment */
    ALLTOALL_CE_STAGE_SETUP, /*< Wait for memhandle setup to finish */
    ALLTOALL_CE_STAGE_POST_COPIES, /*< Post copies */
    ALLTOALL_CE_STAGE_COPY, /*< Wait for all copies to finish */
    ALLTOALL_CE_STAGE_BAR,  /*< Wait for other ranks to finish */
};

//NOLINTNEXTLINE(misc-unused-parameters): stream parameter unused as executor manages execution
ucc_status_t ee_copy_post(void *dst, void *src, size_t len,
                       ucc_ee_executor_t       *executor,
                       ucc_ee_executor_task_t **task, cudaStream_t stream)
{
    (void)stream; /* Unused parameter */
    ucc_ee_executor_task_args_t exec_args = {0};
    exec_args.task_type                   = UCC_EE_EXECUTOR_TASK_COPY;
    exec_args.copy.dst                    = dst;
    exec_args.copy.src                    = src;
    exec_args.copy.len                    = len;
    return ucc_ee_executor_task_post(executor, &exec_args, task);
}

//NOLINTNEXTLINE(misc-unused-parameters): executor and task unused as operation handled by CUDA
ucc_status_t cuda_copy_post(void *dst, void *src, size_t len,
                       ucc_ee_executor_t       *executor,
                       ucc_ee_executor_task_t **task, cudaStream_t stream)
{
    (void)executor; /* Unused parameter */
    (void)task;     /* Unused parameter */
    ucc_status_t status;
    CUDA_CHECK_GOTO(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, stream),
        exit_err, status);
    return UCC_OK;
exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    int i;

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);

    // Clean up completion events
    if (task->alltoallv_ce.evtCompletions) {
        for (i = 0; i < team->num_streams; i++) {
            if (task->alltoallv_ce.evtCompletions[i]) {
                cudaEventDestroy(task->alltoallv_ce.evtCompletions[i]);
            }
        }
        ucc_free(task->alltoallv_ce.evtCompletions);
        task->alltoallv_ce.evtCompletions = NULL;
    }
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_alltoallv_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    ucc_status_t        status;
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_ee_h            ee      = task->super.ee;
    cudaStream_t        stream  = (ee) ? (cudaStream_t)ee->ee_context : UCC_TL_CUDA_TEAM_STREAM_IDX(team, 0);

    // For Alltoallv: copy counts and displ. to SHM for remote GPUs to access (if required)
    if (UCC_COLL_TYPE_ALLTOALLV == args->coll_type) {
        int    i;
        size_t rdt_size = ucc_dt_size(task->alltoallv_ce.rdt);
        size_t sdt_size = ucc_dt_size(task->alltoallv_ce.sdt);
        for (i = 0; i < UCC_TL_TEAM_SIZE(team); ++i) {
            sync->alltoallv_ce.sbytes[i] =
                sdt_size * (size_t)ucc_coll_args_get_count(
                               args, task->alltoallv_ce.scnts, i);
            sync->alltoallv_ce.rbytes[i] =
                rdt_size * (size_t)ucc_coll_args_get_count(
                               args, task->alltoallv_ce.rcnts, i);
            sync->alltoallv_ce.sdispl_bytes[i] =
                sdt_size * (size_t)ucc_coll_args_get_displacement(
                               args, task->alltoallv_ce.sdispl, i);
            sync->alltoallv_ce.rdispl_bytes[i] =
                rdt_size * (size_t)ucc_coll_args_get_displacement(
                               args, task->alltoallv_ce.rdispl, i);
        }
    }
    memcpy(&sync->mem_info_src, &task->alltoallv_ce.mem_info_src,
           sizeof(ucc_tl_cuda_mem_info_t));
    memcpy(&sync->mem_info_dst, &task->alltoallv_ce.mem_info_dst,
           sizeof(ucc_tl_cuda_mem_info_t));
    CUDA_CHECK_GOTO(cudaEventRecord(sync->ipc_event_local, stream), exit_err,
                    status);
    ucc_memory_cpu_store_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_alltoallv_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t          *team = TASK_TEAM(task);
    volatile ucc_tl_cuda_sync_t *peer_sync, *sync;
    ucc_tl_cuda_cache_t         *cache;
    ucc_status_t                 status;
    ucc_rank_t                   i, dst;
    ucc_ee_h                     ee     = task->super.ee;
    cudaStream_t                 stream = (ee) ? (cudaStream_t)ee->ee_context : TASK_STREAM(task);

    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (status != UCC_OK) {
        return status;
    }

    sync = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == UCC_TL_TEAM_RANK(team) ||
            !ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo,
                                             UCC_TL_TEAM_RANK(team), i)) {
            continue;
        }
        peer_sync = TASK_SYNC(task, i);
        cache     = ucc_tl_cuda_get_cache(team, i);
        if (ucc_unlikely(!cache)) {
            status = UCC_ERR_NO_MESSAGE;
            goto exit_err;
        }
        status = ucc_tl_cuda_map_memhandle(
            peer_sync->mem_info_src.ptr, peer_sync->mem_info_src.length,
            peer_sync->mem_info_src.handle,
            &task->alltoallv_ce.peer_map_addr_src[i], cache);
        if (UCC_OK != status) {
            ucc_error("ucc_cuda_ipc_map_memhandle failed");
            return UCC_ERR_INVALID_PARAM;
        }

        CUDA_CHECK_GOTO(
            cudaStreamWaitEvent(stream, sync->data[i].ipc_event_remote, 0),
            exit_err, status);
    }

    for (i = 0; i < team->topo->num_proxies; i++) {
        dst = team->topo->proxies[i].dst;
        peer_sync = TASK_SYNC(task, dst);
        cache     = ucc_tl_cuda_get_cache(team, dst);
        if (ucc_unlikely(!cache)) {
            status = UCC_ERR_NO_MESSAGE;
            goto exit_err;
        }
        status = ucc_tl_cuda_map_memhandle(
            peer_sync->mem_info_dst.ptr, peer_sync->mem_info_dst.length,
            peer_sync->mem_info_dst.handle,
            &task->alltoallv_ce.peer_map_addr_dst[dst], cache);
        if (UCC_OK != status) {
            ucc_error("ucc_cuda_ipc_map_memhandle failed");
            return UCC_ERR_INVALID_PARAM;
        }
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_post_copies(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team        = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib         = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_rank_t          rank        = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_sync_t *sync        = TASK_SYNC(task, rank);
    ucc_ee_h            ee          = task->super.ee;
    ucc_status_t        status      = UCC_OK;
    cudaStream_t        stream      = 0;
    int                 stream_idx  = 0;
    int                 num_streams = UCC_TL_CUDA_TEAM_NUM_STREAMS(team);
    ucc_tl_cuda_sync_t *peer_sync;
    ucc_ee_executor_t  *exec;
    void               *src, *dst;
    size_t              data_size, data_displ;
    ucc_rank_t          i, peer, psrc, pdst;

    if (lib->cfg.alltoall_use_copy_engine) {
        // If triggered post, use the stream from the executor
        if (ee) {
            stream = (cudaStream_t)ee->ee_context;
            num_streams = 1;
            stream_idx = 0;
        }
        // copy engine is used, so no executor is needed
        exec = NULL;
    } else {
        stream = 0;
        status = ucc_coll_task_get_executor(&task->super, &exec);
        if (ucc_unlikely(status != UCC_OK)) {
            goto exit;
        }
    }

    task->alltoallv_ce.num_posted = 0;

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        peer = (rank + i) % UCC_TL_TEAM_SIZE(team);
        if (!ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo, rank,
                                             peer)) {
            continue;
        }
        peer_sync = TASK_SYNC(task, peer);
        if (peer == rank) {
            src = task->alltoallv_ce.sbuf;
        } else {
            src = PTR_OFFSET(task->alltoallv_ce.peer_map_addr_src[peer],
                             peer_sync->mem_info_src.offset);
        }
        data_size = task->alltoallv_ce.get_size(
            task, peer_sync->alltoallv_ce.sbytes, rank);
        if (data_size == 0) {
            continue;
        }
        data_displ = task->alltoallv_ce.get_offset(
            task, peer_sync->alltoallv_ce.sdispl_bytes, rank);
        src        = PTR_OFFSET(src, data_displ);
        data_displ = task->alltoallv_ce.get_offset(
            task, sync->alltoallv_ce.rdispl_bytes, peer);
        dst = PTR_OFFSET(task->alltoallv_ce.rbuf, data_displ);

        // If triggered post, use the stream from the executor
        if (lib->cfg.alltoall_use_copy_engine && !ee) {
            // Get the current stream
            stream = UCC_TL_CUDA_TEAM_STREAM_IDX(team, stream_idx);
            // Round-robin across available streams
            ucc_assume(num_streams > 0);
            stream_idx = (stream_idx + 1) % num_streams;
        }

        status = task->alltoallv_ce.copy_post(
            dst, src, data_size, exec,
            &task->alltoallv_ce.exec_task[task->alltoallv_ce.num_posted],
            stream);

        if (ucc_unlikely(status != UCC_OK)) {
            goto exit;
        }
        task->alltoallv_ce.num_posted++;
    }

    for (i = 0; i < team->topo->num_proxies; i++) {
        psrc      = team->topo->proxies[i].src;
        pdst      = team->topo->proxies[i].dst;
        peer_sync = TASK_SYNC(task, psrc);
        data_size = task->alltoallv_ce.get_size(
            task, peer_sync->alltoallv_ce.sbytes, pdst);
        if (data_size == 0) {
            continue;
        }
        data_displ = task->alltoallv_ce.get_offset(
            task, peer_sync->alltoallv_ce.sdispl_bytes, pdst);
        src        = PTR_OFFSET(task->alltoallv_ce.peer_map_addr_src[psrc],
                                peer_sync->mem_info_src.offset);
        src        = PTR_OFFSET(src, data_displ);
        peer_sync  = TASK_SYNC(task, pdst);
        dst        = PTR_OFFSET(task->alltoallv_ce.peer_map_addr_dst[pdst],
                                peer_sync->mem_info_dst.offset);
        data_displ = task->alltoallv_ce.get_offset(
            task, peer_sync->alltoallv_ce.rdispl_bytes, psrc);
        dst                 = PTR_OFFSET(dst, data_displ);

        // If triggered post, use the stream from the executor
        if (lib->cfg.alltoall_use_copy_engine && !ee) {
            // Get the current stream
            stream = team->streams[stream_idx];
            // Round-robin across available streams
            ucc_assume(num_streams > 0);
            stream_idx = (stream_idx + 1) % num_streams;
        }

        status = task->alltoallv_ce.copy_post(
            dst, src, data_size, exec,
            &task->alltoallv_ce.exec_task[task->alltoallv_ce.num_posted],
            stream);

        if (ucc_unlikely(status != UCC_OK)) {
            goto exit;
        }
        task->alltoallv_ce.num_posted++;
    }

    if (lib->cfg.alltoall_use_copy_engine) {
        if (ee) {
            CUDA_CHECK_GOTO(
                cudaEventRecord(task->alltoallv_ce.evtCompletions[0], stream),
                exit, status);
        } else {
            // Record completion events for each stream
            for (i = 0; i < num_streams; i++) {
                CUDA_CHECK_GOTO(
                    cudaEventRecord(task->alltoallv_ce.evtCompletions[i],
                                    UCC_TL_CUDA_TEAM_STREAM_IDX(team, i)),
                    exit, status);
            }
        }
    }

exit:
    return status;
}

#if CUDART_VERSION >= 12080
ucc_status_t ucc_tl_cuda_alltoallv_ce_post_batch_copies(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team           = TASK_TEAM(task);
    ucc_rank_t          rank           = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_sync_t *sync           = TASK_SYNC(task, rank);
    ucc_ee_h            ee             = task->super.ee;
    ucc_status_t        status         = UCC_OK;
    cudaStream_t        stream         = 0;
    int                 stream_idx     = 0;
    size_t              op_count       = 0;
    void              **srcs           = NULL;
    void              **dsts           = NULL;
    size_t             *sizes          = NULL;
    struct cudaMemcpyAttributes *attrs = NULL;
    size_t             *attrsIdxs      = NULL;
    ucc_rank_t          i, peer;
    void               *src, *dst;
    size_t              data_size, data_displ;
    size_t              fail_idx;

    // Count total operations
    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        peer = (rank + i) % UCC_TL_TEAM_SIZE(team);
        ucc_tl_cuda_sync_t *peer_sync = TASK_SYNC(task, peer);
        data_size = task->alltoallv_ce.get_size(task, peer_sync->alltoallv_ce.sbytes, rank);
        if (data_size > 0) {
            op_count++;
        }
    }

    // If there is nothing to copy, just record completion and return
    if (op_count == 0) {
        if (ee) {
            stream = (cudaStream_t)ee->ee_context;
            CUDA_CHECK_GOTO(
                cudaEventRecord(task->alltoallv_ce.evtCompletions[0], stream),
                exit, status);
        } else {
            stream = UCC_TL_CUDA_TEAM_STREAM_IDX(team, stream_idx);
            CUDA_CHECK_GOTO(
                cudaEventRecord(task->alltoallv_ce.evtCompletions[stream_idx], stream),
                exit, status);
        }
        task->alltoallv_ce.num_posted = 0;
        goto exit;
    }

    // Allocate arrays
    srcs = ucc_malloc(op_count * sizeof(void *), "srcs");
    dsts = ucc_malloc(op_count * sizeof(void *), "dsts");
    sizes = ucc_malloc(op_count * sizeof(size_t), "sizes");
    attrs = ucc_malloc(op_count * sizeof(struct cudaMemcpyAttributes), "attrs");
    attrsIdxs = ucc_malloc(op_count * sizeof(size_t), "attrsIdxs");

    if (!srcs || !dsts || !sizes || !attrs || !attrsIdxs) {
        status = UCC_ERR_NO_MEMORY;
        goto exit;
    }

    if (ee) {
        stream = (cudaStream_t)ee->ee_context;
    } else {
        stream = UCC_TL_CUDA_TEAM_STREAM_IDX(team, stream_idx);
    }
    op_count = 0;

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        peer = (rank + i) % UCC_TL_TEAM_SIZE(team);

        ucc_tl_cuda_sync_t *peer_sync = TASK_SYNC(task, peer);

        data_size = task->alltoallv_ce.get_size(task, peer_sync->alltoallv_ce.sbytes, rank);
        if (data_size == 0) {
            continue;
        }

        // Source buffer
        if (peer == rank) {
            src = task->alltoallv_ce.sbuf;
        } else {
            src = PTR_OFFSET(task->alltoallv_ce.peer_map_addr_src[peer],
                             peer_sync->mem_info_src.offset);
        }

        data_displ = task->alltoallv_ce.get_offset(
            task, peer_sync->alltoallv_ce.sdispl_bytes, rank);
        src = PTR_OFFSET(src, data_displ);

        // Destination buffer
        data_displ = task->alltoallv_ce.get_offset(
            task, sync->alltoallv_ce.rdispl_bytes, peer);
        dst = PTR_OFFSET(task->alltoallv_ce.rbuf, data_displ);

        srcs[op_count] = src;
        dsts[op_count] = dst;
        sizes[op_count] = data_size;

        // Setup attributes: leave location hints unset so runtime infers devices
        memset(&attrs[op_count], 0, sizeof(struct cudaMemcpyAttributes));
        attrs[op_count].srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attrs[op_count].flags = cudaMemcpyFlagPreferOverlapWithCompute;

        attrsIdxs[op_count] = op_count;

        op_count++;
    }

    // Launch batch copy
    cudaError_t cerr = cudaMemcpyBatchAsync(
        dsts, srcs, sizes, op_count,
        attrs, attrsIdxs, op_count,
        &fail_idx, stream);

    if (cerr != cudaSuccess) {
        if (fail_idx != SIZE_MAX) {
            ucc_error("cudaMemcpyBatchAsync failed: %s (failIdx=%zu)",
                      cudaGetErrorString(cerr), fail_idx);
        } else {
            ucc_error("cudaMemcpyBatchAsync failed: %s",
                      cudaGetErrorString(cerr));
        }
        status = UCC_ERR_NO_MESSAGE;
        goto exit;
    }

    task->alltoallv_ce.num_posted = op_count;

    // Record completion events
    if (ee) {
        CUDA_CHECK_GOTO(
            cudaEventRecord(task->alltoallv_ce.evtCompletions[0], stream),
            exit, status);
    } else {
        CUDA_CHECK_GOTO(
            cudaEventRecord(task->alltoallv_ce.evtCompletions[stream_idx], stream),
            exit, status);
    }

exit:
    ucc_free(srcs);
    ucc_free(dsts);
    ucc_free(sizes);
    ucc_free(attrs);
    ucc_free(attrsIdxs);
    return status;
}
#endif

ucc_status_t ucc_tl_cuda_alltoallv_unmap(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_rank_t                   i, dst;
    volatile ucc_tl_cuda_sync_t *peer_sync;
    ucc_tl_cuda_cache_t         *cache;
    ucc_status_t                 status;

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == UCC_TL_TEAM_RANK(team) ||
            !ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo,
                                            UCC_TL_TEAM_RANK(team), i)) {
            continue;
        }
        peer_sync = TASK_SYNC(task, i);
        cache     = ucc_tl_cuda_get_cache(team, i);

        status = ucc_tl_cuda_unmap_memhandle(
            (uintptr_t)peer_sync->mem_info_src.ptr,
            task->alltoallv_ce.peer_map_addr_src[i], cache, 0);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    }

    for (i = 0; i < team->topo->num_proxies; i++) {
        dst = team->topo->proxies[i].dst;
        peer_sync = TASK_SYNC(task, dst);
        cache = ucc_tl_cuda_get_cache(team, dst);

        status = ucc_tl_cuda_unmap_memhandle(
            (uintptr_t)peer_sync->mem_info_dst.ptr,
            task->alltoallv_ce.peer_map_addr_dst[dst], cache, 0);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    }

    return UCC_OK;
}

void ucc_tl_cuda_alltoallv_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib  = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_ee_h            ee   = task->super.ee;
    ucc_status_t        status;
    int                 i;

    switch (task->alltoallv_ce.stage) {
    case ALLTOALL_CE_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        status = ucc_tl_cuda_alltoallv_setup_start(task);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_SETUP;
        /* fall through */
    case ALLTOALL_CE_STAGE_SETUP:
        status = ucc_tl_cuda_alltoallv_setup_test(task);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        /* fall through */
    case ALLTOALL_CE_STAGE_POST_COPIES:
        /* Use batch copies only on CUDA 12.8+; otherwise fall back */
#if CUDART_VERSION >= 12080
        status = ucc_tl_cuda_alltoallv_ce_post_batch_copies(task);
#else
        status = ucc_tl_cuda_alltoallv_ce_post_copies(task);
#endif
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_COPY;
        /* fall through */
    case ALLTOALL_CE_STAGE_COPY:
        if (lib->cfg.alltoall_use_copy_engine) {
            int all_completed = 1;
            int num_streams = (ee) ? 1 : team->num_streams;
            for (i = 0; i < num_streams; i++) {
                cudaError_t cuda_status =
                    cudaEventQuery(task->alltoallv_ce.evtCompletions[i]);
                if (cuda_status == cudaSuccess) {
                    // This event is completed
                    continue;
                } else if (cuda_status == cudaErrorNotReady) {
                    // This event is still in progress
                    all_completed = 0;
                    break;
                } else {
                    // Error occurred
                    ucc_error("error cudaEventQuery %s!",
                              cudaGetErrorString(cuda_status));
                    task->super.status = UCC_ERR_NO_MESSAGE;
                    ucc_assert(0);
                    return;
                }
            }

            if (all_completed) {
                ucc_debug("all cuda copies finished");
                task->super.status = UCC_OK;
            } else {
                task->super.status = UCC_INPROGRESS;
                return;
            }
        } else {
            for (i = 0; i < task->alltoallv_ce.num_posted; i++) {
                if (!task->alltoallv_ce.exec_task[i]) {
                    continue;
                }
                status =
                    ucc_ee_executor_task_test(task->alltoallv_ce.exec_task[i]);
                if (status != UCC_OK) {
                    if (status == UCC_OPERATION_INITIALIZED) {
                        status = UCC_INPROGRESS;
                    }
                    task->super.status = status;
                    return;
                }
                ucc_ee_executor_task_finalize(task->alltoallv_ce.exec_task[i]);
                task->alltoallv_ce.exec_task[i] = NULL;
            }
        }

        status =
            ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_BAR;
        /* fall through */
    default:
        ucc_assert(task->alltoallv_ce.stage == ALLTOALL_CE_STAGE_BAR);
        break;
    }

    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (status == UCC_OK) {
        status = ucc_tl_cuda_alltoallv_unmap(task);
        ucc_tl_cuda_put_sync(task);
    }

    task->super.status = status;
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    // if not triggered post or copy engine is used, we need to start from sync
    if (task->alltoallv_ce.stage != ALLTOALL_CE_STAGE_POST_COPIES && task->alltoallv_ce.stage != ALLTOALL_CE_STAGE_COPY) {
        task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_SYNC;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t
ucc_tl_cuda_alltoallv_ce_triggered_post_setup(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_status_t        status;

    do {
        status = ucc_tl_cuda_get_sync(task);
    } while (status == UCC_INPROGRESS);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_tl_cuda_alltoallv_setup_start(task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_cuda_put_sync(task);
        return status;
    }

    do {
        status = ucc_tl_cuda_alltoallv_setup_test(task);
    } while (status == UCC_INPROGRESS);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_cuda_put_sync(task);
        return status;
    }
    task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_POST_COPIES;

    return UCC_OK;
}

//NOLINTNEXTLINE: task is unused
size_t ucc_tl_cuda_alltoallv_get_size(const ucc_tl_cuda_task_t *task,
                                      size_t *sizes, ucc_rank_t block)
{
    return sizes[block];
}

//NOLINTNEXTLINE: task is unused
size_t ucc_tl_cuda_alltoallv_get_offset(const ucc_tl_cuda_task_t *task,
                                        size_t *displ, ucc_rank_t block)
{
    return displ[block];
}

//NOLINTNEXTLINE(misc-unused-parameters): ev parameter unused as it's not needed for this implementation
ucc_status_t ucc_tl_cuda_alltoallv_ce_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                                     ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_status_t        status;
    ucc_ev_t            post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_debug(UCC_TASK_LIB(task), "triggered post. task:%p", coll_task);
    coll_task->triggered_post_setup(coll_task);


    /* Use batch copies only on CUDA 12.8+; otherwise fall back */
#if CUDART_VERSION >= 12080
    status = ucc_tl_cuda_alltoallv_ce_post_batch_copies(task);
#else
    status = ucc_tl_cuda_alltoallv_ce_post_copies(task);
#endif
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to post copies\n");
        return status;
    }
    task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_COPY;

    status = coll_task->post(coll_task);
    if (ucc_likely(status == UCC_OK)) {
        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.ev_context      = NULL;
        post_event.req             = &coll_task->super;
        ucc_ee_set_event_internal(coll_task->ee, &post_event,
                                  &coll_task->ee->event_out_queue);
    }
    return status;
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib  = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_status_t        status;
    size_t              data_len;
    int                 i;

    if (!UCC_COLL_ARGS_CONTIG_BUFFER(args)) {
        tl_debug(UCC_TL_TEAM_LIB(team), "Do not support non-contiguous buffer");
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->alltoallv_ce.get_size   = ucc_tl_cuda_alltoallv_get_size;
    task->alltoallv_ce.get_offset = ucc_tl_cuda_alltoallv_get_offset;
    task->alltoallv_ce.sdt        = args->src.info_v.datatype;
    task->alltoallv_ce.rdt        = args->dst.info_v.datatype;
    task->alltoallv_ce.sbuf       = args->src.info_v.buffer;
    task->alltoallv_ce.rbuf       = args->dst.info_v.buffer;
    task->alltoallv_ce.scnts      = args->src.info_v.counts;
    task->alltoallv_ce.rcnts      = args->dst.info_v.counts;
    task->alltoallv_ce.sdispl     = args->src.info_v.displacements;
    task->alltoallv_ce.rdispl     = args->dst.info_v.displacements;
    task->alltoallv_ce.stage      = ALLTOALL_CE_STAGE_SYNC;

    data_len = ucc_dt_size(args->src.info_v.datatype) *
               ucc_coll_args_get_total_count(args, args->src.info_v.counts,
                                             UCC_TL_TEAM_SIZE(team));
    status = ucc_tl_cuda_mem_info_get(args->src.info_v.buffer, data_len,
                                      &task->alltoallv_ce.mem_info_src);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    if (team->topo->proxy_needed) {
        data_len = ucc_dt_size(args->dst.info_v.datatype) *
                ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                              UCC_TL_TEAM_SIZE(team));
        status = ucc_tl_cuda_mem_info_get(args->dst.info_v.buffer, data_len,
                                          &task->alltoallv_ce.mem_info_dst);
        if (ucc_unlikely(status != UCC_OK)) {
            goto exit_err;
        }
    }

    if (lib->cfg.alltoall_use_copy_engine) {
        ucc_debug("ucc_tl_cuda_alltoallv_ce_init: copy engine");
        task->super.triggered_post = ucc_tl_cuda_alltoallv_ce_triggered_post;

        task->alltoallv_ce.copy_post = cuda_copy_post;
        task->alltoallv_ce.evtCompletions = (cudaEvent_t*)ucc_malloc(team->num_streams * sizeof(cudaEvent_t), "alltoallv_ce.evtCompletions");

        for (i = 0; i < team->num_streams; i++) {
            CUDA_CHECK_GOTO(cudaEventCreateWithFlags(&task->alltoallv_ce.evtCompletions[i], cudaEventDisableTiming), exit_err, status);
        }
    } else {
        ucc_debug("ucc_tl_cuda_alltoallv_ce_init: executor");
        task->alltoallv_ce.copy_post = ee_copy_post;
        task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    }

    task->super.post = ucc_tl_cuda_alltoallv_ce_start;
    task->super.triggered_post_setup =
        ucc_tl_cuda_alltoallv_ce_triggered_post_setup;

    task->super.progress = ucc_tl_cuda_alltoallv_ce_progress;
    task->super.finalize = ucc_tl_cuda_alltoallv_ce_finalize;
    task->bar            = TASK_BAR(task);
    return UCC_OK;

exit_err:
    return status;
}
