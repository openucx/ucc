/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

enum {
    ALLTOALL_CE_STAGE_SYNC,  /*< Wait for free SYNC segment */
    ALLTOALL_CE_STAGE_SETUP, /*< Wait for memhandle setup to finish */
    ALLTOALL_CE_STAGE_POST_COPIES,
    ALLTOALL_CE_STAGE_COPY, /*< Wait for all copies to finish */
    ALLTOALL_CE_STAGE_BAR,  /*< Wait for other ranks to finish */
};

ucc_status_t ucc_tl_cuda_alltoallv_ce_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
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
    cudaStream_t        stream  = (ee) ? (cudaStream_t)ee->ee_context : team->stream;

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
    ucc_rank_t                   i;
    ucc_ee_h                     ee     = task->super.ee;
    cudaStream_t                 stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;

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
        status = ucc_tl_cuda_map_memhandle(
            peer_sync->mem_info_dst.ptr, peer_sync->mem_info_dst.length,
            peer_sync->mem_info_dst.handle,
            &task->alltoallv_ce.peer_map_addr_dst[i], cache);
        if (UCC_OK != status) {
            ucc_error("ucc_cuda_ipc_map_memhandle failed");
            return UCC_ERR_INVALID_PARAM;
        }
        CUDA_CHECK_GOTO(
            cudaStreamWaitEvent(stream, sync->data[i].ipc_event_remote, 0),
            exit_err, status);
    }
    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_post_copies(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t         *team = TASK_TEAM(task);
    ucc_rank_t                  rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_sync_t         *sync = TASK_SYNC(task, rank);
    ucc_tl_cuda_sync_t         *peer_sync;
    ucc_ee_executor_t          *exec;
    void                       *src, *dst;
    ucc_ee_executor_task_t    **exec_task;
    size_t                      data_size, data_displ;
    ucc_rank_t                  i, peer, psrc, pdst;
    ucc_status_t                status;
    ucc_ee_executor_task_args_t exec_args;

    task->alltoallv_ce.num_posted = 0;
    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit;
    }

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
        exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
        exec_args.bufs[0]   = dst;
        exec_args.bufs[1]   = src;
        exec_args.count     = data_size;
        exec_task =
            &task->alltoallv_ce.exec_task[task->alltoallv_ce.num_posted];
        status = ucc_ee_executor_task_post(exec, &exec_args, exec_task);
        if (ucc_unlikely(status != UCC_OK)) {
            goto exit;
        }
        task->alltoallv_ce.num_posted++;
    }

    for (i = 0; i < team->topo->num_proxies; i++) {
        if (team->topo->proxies[i].proxy == rank) {
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
            dst        = PTR_OFFSET(dst, data_displ);

            exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
            exec_args.bufs[0]   = dst;
            exec_args.bufs[1]   = src;
            exec_args.count     = data_size;
            exec_task =
                &task->alltoallv_ce.exec_task[task->alltoallv_ce.num_posted];
            status = ucc_ee_executor_task_post(exec, &exec_args, exec_task);
            if (ucc_unlikely(status != UCC_OK)) {
                goto exit;
            }
            task->alltoallv_ce.num_posted++;
        }
    }
exit:
    return status;
}

void ucc_tl_cuda_alltoallv_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
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
    case ALLTOALL_CE_STAGE_SETUP:
        status = ucc_tl_cuda_alltoallv_setup_test(task);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
    case ALLTOALL_CE_STAGE_POST_COPIES:
        status = ucc_tl_cuda_alltoallv_ce_post_copies(task);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_COPY;
    case ALLTOALL_CE_STAGE_COPY:
        for (i = 0; i < task->alltoallv_ce.num_posted; i++) {
            if (!task->alltoallv_ce.exec_task[i]) {
                continue;
            }
            status = ucc_ee_executor_task_test(task->alltoallv_ce.exec_task[i]);
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
        status =
            ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->alltoallv_ce.stage = ALLTOALL_CE_STAGE_BAR;
    default:
        ucc_assert(task->alltoallv_ce.stage == ALLTOALL_CE_STAGE_BAR);
        break;
    }
    task->super.status =
        ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (task->super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    if (task->alltoallv_ce.stage != ALLTOALL_CE_STAGE_POST_COPIES) {
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

//NOLINTNEXTLINE
size_t ucc_tl_cuda_alltoallv_get_size(const ucc_tl_cuda_task_t *task,
                                      size_t *sizes, ucc_rank_t block)
{
    return sizes[block];
}

size_t ucc_tl_cuda_alltoallv_get_offset(const ucc_tl_cuda_task_t *task,
                                        size_t *displ, ucc_rank_t block)
{
    return displ[block];
}

ucc_status_t ucc_tl_cuda_alltoallv_ce_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_status_t        status;
    size_t              data_len;

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
    data_len = ucc_dt_size(args->dst.info_v.datatype) *
               ucc_coll_args_get_total_count(args, args->dst.info_v.counts,
                                             UCC_TL_TEAM_SIZE(team));
    status = ucc_tl_cuda_mem_info_get(args->dst.info_v.buffer, data_len,
                                      &task->alltoallv_ce.mem_info_dst);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_alltoallv_ce_start;
    task->super.triggered_post = ucc_triggered_post;
    task->super.triggered_post_setup =
        ucc_tl_cuda_alltoallv_ce_triggered_post_setup;
    task->super.progress = ucc_tl_cuda_alltoallv_ce_progress;
    task->super.finalize = ucc_tl_cuda_alltoallv_ce_finalize;
    task->bar            = TASK_BAR(task);

    return UCC_OK;

exit_err:
    return status;
}
