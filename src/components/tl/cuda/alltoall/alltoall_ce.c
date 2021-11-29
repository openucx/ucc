/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"
#include "core/ucc_mc.h"
#include "tl_cuda_cache.h"

enum {
    ALLTOALL_CE_STAGE_SYNC, /*< Wait for free SYNC segment */
    ALLTOALL_CE_STAGE_COPY, /*< Wait for all copies to finish */
    ALLTOALL_CE_STAGE_BAR,  /*< Wait for other ranks to finish */
};

ucc_status_t ucc_tl_cuda_alltoall_ce_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_mc_ee_destroy_event((void*)task->alltoall_ce.copy_done,
                            UCC_EE_CUDA_STREAM);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_alltoall_setup(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t          *team = TASK_TEAM(task);
    ucc_tl_cuda_sync_t          *sync = TASK_SYNC(task, team->rank);
    volatile ucc_tl_cuda_sync_t *peer_sync;
    ucc_tl_cuda_cache_t         *cache;
    ucc_status_t                 status;
    ucc_rank_t                   i;

    sync->ptr = task->alltoall_ce.mem_info.ptr;
    sync->length = task->alltoall_ce.mem_info.length;
    sync->offset = task->alltoall_ce.mem_info.offset;
    memcpy(&sync->handle, &task->alltoall_ce.mem_info.handle,
           sizeof(cudaIpcMemHandle_t));
    CUDACHECK_GOTO(cudaEventRecord(sync->ipc_event_local, team->stream),
                   exit_err, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_cuda_shm_barrier_start(team->rank, task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }
    while (ucc_tl_cuda_shm_barrier_test(team->rank, task->bar) == UCC_INPROGRESS);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    for (i = 0; i < team->size; i++) {
        if (i == team->rank) {
            continue;
        }
        peer_sync = TASK_SYNC(task, i);
        cache = ucc_tl_cuda_get_cache(team, i);
        if (ucc_unlikely(!cache)) {
            status = UCC_ERR_NO_MESSAGE;
            goto exit_err;
        }
        status = ucc_tl_cuda_map_memhandle(peer_sync->ptr, peer_sync->length,
                                           peer_sync->handle,
                                           &task->alltoall_ce.peer_map_addr[i],
                                           cache);
        if (UCC_OK != status) {
            ucc_error("ucc_cuda_ipc_map_memhandle failed");
            return UCC_ERR_INVALID_PARAM;
        }
    }
    return UCC_OK;

exit_err:
    return status;
}

static ucc_status_t ucc_tl_cuda_alltoall_ce_post_copies(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, team->rank);
    ucc_tl_cuda_sync_t *peer_sync;
    size_t send_len;
    ucc_rank_t i, peer;
    void *src, *dst;
    ucc_status_t status;

    send_len = (args->src.info.count / team->size) *
               ucc_dt_size(args->src.info.datatype);
    for (i = 0; i < team->size; i++) {
        peer = (team->rank + i) % team->size;
        peer_sync = TASK_SYNC(task, peer);
        if (peer == team->rank) {
            src = args->src.info.buffer;
        } else {
            src = PTR_OFFSET(task->alltoall_ce.peer_map_addr[peer],
                             peer_sync->offset);
            CUDACHECK_GOTO(cudaStreamWaitEvent(team->stream,
                                               sync->data[peer].ipc_event_remote,
                                               0),
                           exit, status, UCC_TASK_LIB(task));
        }
        src = PTR_OFFSET(src, team->rank * send_len);
        dst = PTR_OFFSET(args->dst.info.buffer, peer * send_len);
        CUDACHECK_GOTO(cudaMemcpyAsync(dst, src, send_len,
                                       cudaMemcpyDeviceToDevice, team->stream),
                       exit, status, UCC_TASK_LIB(task));
    }

    status = ucc_mc_ee_event_post(team->stream, task->alltoall_ce.copy_done,
                                  UCC_EE_CUDA_STREAM);
exit:
    return status;

}

ucc_status_t ucc_tl_cuda_alltoall_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, team->rank);
    ucc_status_t        status;

    if (task->alltoall_ce.stage == ALLTOALL_CE_STAGE_SYNC) {
        if (sync->seq_num[0] != task->seq_num) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }
        status = ucc_tl_cuda_alltoall_setup(task);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        status = ucc_tl_cuda_alltoall_ce_post_copies(task);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        task->alltoall_ce.stage = ALLTOALL_CE_STAGE_COPY;
    }

    if(task->alltoall_ce.stage == ALLTOALL_CE_STAGE_COPY) {
        status = ucc_mc_ee_event_test(task->alltoall_ce.copy_done,
                                      UCC_EE_CUDA_STREAM);
        if (status != UCC_OK) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        status = ucc_tl_cuda_shm_barrier_start(team->rank, task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        task->alltoall_ce.stage = ALLTOALL_CE_STAGE_BAR;
    }
    task->super.super.status = ucc_tl_cuda_shm_barrier_test(team->rank,
                                                            task->bar);
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_alltoall_ce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    task->alltoall_ce.stage = ALLTOALL_CE_STAGE_SYNC;
    ucc_tl_cuda_alltoall_ce_progress(coll_task);
    if (task->super.super.status == UCC_INPROGRESS) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_cuda_alltoall_ce_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_status_t status;
    size_t data_len;

    status = ucc_mc_ee_create_event(&task->alltoall_ce.copy_done,
                                    UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    data_len = ucc_dt_size(args->src.info.datatype) * args->src.info.count;
    status = ucc_tl_cuda_mem_info_get(args->src.info.buffer, data_len, team,
                                      &task->alltoall_ce.mem_info);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    task->super.post     = ucc_tl_cuda_alltoall_ce_start;
    task->super.progress = ucc_tl_cuda_alltoall_ce_progress;
    task->super.finalize = ucc_tl_cuda_alltoall_ce_finalize;
    task->bar            = TASK_BAR(task);

    return UCC_OK;

exit_err:
    ucc_mc_ee_destroy_event(task->alltoall_ce.copy_done, UCC_MEMORY_TYPE_CUDA);
    return status;
}
