/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

enum {
    ALLTOALL_CE_STAGE_SYNC,  /*< Wait for free SYNC segment */
    ALLTOALL_CE_STAGE_SETUP, /*< Wait for memhandle setup to finish */
    ALLTOALL_CE_STAGE_COPY,  /*< Wait for all copies to finish */
    ALLTOALL_CE_STAGE_BAR,   /*< Wait for other ranks to finish */
};

ucc_status_t ucc_tl_cuda_alltoall_ce_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_ec_destroy_event((void*)task->alltoall_ce.copy_done,
                         UCC_EE_CUDA_STREAM);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_alltoall_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t          *team = TASK_TEAM(task);
    ucc_tl_cuda_sync_t          *sync = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    ucc_status_t                 status;

    memcpy(&sync->mem_info_src, &task->alltoall_ce.mem_info_src,
           sizeof(ucc_tl_cuda_mem_info_t));
    memcpy(&sync->mem_info_dst, &task->alltoall_ce.mem_info_dst,
           sizeof(ucc_tl_cuda_mem_info_t));
    CUDA_CHECK_GOTO(cudaEventRecord(sync->ipc_event_local, team->stream),
                    exit_err, status);
    ucc_memory_bus_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_alltoall_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t          *team = TASK_TEAM(task);
    volatile ucc_tl_cuda_sync_t *peer_sync;
    ucc_tl_cuda_cache_t         *cache;
    ucc_status_t                 status;
    ucc_rank_t                   i;

    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (status != UCC_OK) {
        return status;
    }

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == UCC_TL_TEAM_RANK(team) ||
            !ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo,
                                             UCC_TL_TEAM_RANK(team), i)) {
            continue;
        }
        peer_sync = TASK_SYNC(task, i);
        cache = ucc_tl_cuda_get_cache(team, i);
        if (ucc_unlikely(!cache)) {
            status = UCC_ERR_NO_MESSAGE;
            goto exit_err;
        }
        status = ucc_tl_cuda_map_memhandle(peer_sync->mem_info_src.ptr,
                                           peer_sync->mem_info_src.length,
                                           peer_sync->mem_info_src.handle,
                                           &task->alltoall_ce.peer_map_addr_src[i],
                                           cache);
        if (UCC_OK != status) {
            ucc_error("ucc_cuda_ipc_map_memhandle failed");
            return UCC_ERR_INVALID_PARAM;
        }
        status = ucc_tl_cuda_map_memhandle(peer_sync->mem_info_dst.ptr,
                                           peer_sync->mem_info_dst.length,
                                           peer_sync->mem_info_dst.handle,
                                           &task->alltoall_ce.peer_map_addr_dst[i],
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
    ucc_rank_t          rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);
    ucc_tl_cuda_sync_t *peer_sync;
    size_t send_len;
    ucc_rank_t i, peer, psrc, pdst;
    void *src, *dst;
    ucc_status_t status;

    send_len = (args->src.info.count / UCC_TL_TEAM_SIZE(team)) *
               ucc_dt_size(args->src.info.datatype);
    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        peer = (rank + i) % UCC_TL_TEAM_SIZE(team);
        if (!ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo,
                                             rank, peer)) {
            continue;
        }
        peer_sync = TASK_SYNC(task, peer);
        if (peer == rank) {
            src = args->src.info.buffer;
        } else {
            src = PTR_OFFSET(task->alltoall_ce.peer_map_addr_src[peer],
                             peer_sync->mem_info_src.offset);
            CUDA_CHECK_GOTO(cudaStreamWaitEvent(team->stream,
                                                sync->data[peer].ipc_event_remote,
                                                0),
                           exit, status);
        }
        src = PTR_OFFSET(src, rank * send_len);
        dst = PTR_OFFSET(args->dst.info.buffer, peer * send_len);
        CUDA_CHECK_GOTO(cudaMemcpyAsync(dst, src, send_len,
                                       cudaMemcpyDeviceToDevice, team->stream),
                       exit, status);
    }

    for (i = 0; i < team->topo->num_proxies; i++) {
        if (team->topo->proxies[i].proxy == rank) {
            psrc = team->topo->proxies[i].src;
            pdst = team->topo->proxies[i].dst;
            peer_sync = TASK_SYNC(task, psrc);
            src = PTR_OFFSET(task->alltoall_ce.peer_map_addr_src[psrc],
                             peer_sync->mem_info_src.offset);
            src = PTR_OFFSET(src, pdst * send_len);
            peer_sync = TASK_SYNC(task, pdst);
            dst = PTR_OFFSET(task->alltoall_ce.peer_map_addr_dst[pdst],
                             peer_sync->mem_info_dst.offset);
            dst = PTR_OFFSET(dst, psrc * send_len);
            CUDA_CHECK_GOTO(cudaMemcpyAsync(dst, src, send_len,
                                            cudaMemcpyDeviceToDevice,
                                            team->stream),
                            exit, status);
        }
    }
    status = ucc_ec_event_post(team->stream, task->alltoall_ce.copy_done,
                               UCC_EE_CUDA_STREAM);
exit:
    return status;

}

ucc_status_t ucc_tl_cuda_alltoall_ce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_status_t        status;

    switch (task->alltoall_ce.stage)
    {
    case ALLTOALL_CE_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }
        status = ucc_tl_cuda_alltoall_setup_start(task);
        if (status != UCC_OK) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        task->alltoall_ce.stage = ALLTOALL_CE_STAGE_SETUP;
    case ALLTOALL_CE_STAGE_SETUP:
        status =  ucc_tl_cuda_alltoall_setup_test(task);
        if (status != UCC_OK) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        status = ucc_tl_cuda_alltoall_ce_post_copies(task);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        task->alltoall_ce.stage = ALLTOALL_CE_STAGE_COPY;
    case ALLTOALL_CE_STAGE_COPY:
        status = ucc_ec_event_test(task->alltoall_ce.copy_done,
                                   UCC_EE_CUDA_STREAM);
        if (status != UCC_OK) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.super.status = status;
            return task->super.super.status;
        }
        task->alltoall_ce.stage = ALLTOALL_CE_STAGE_BAR;
    default:
        ucc_assert(task->alltoall_ce.stage == ALLTOALL_CE_STAGE_BAR);
        break;
    }
    task->super.super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                            task->bar);
    if (task->super.super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
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

    status = ucc_ec_create_event(&task->alltoall_ce.copy_done,
                                 UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    data_len = ucc_dt_size(args->src.info.datatype) * args->src.info.count;
    status = ucc_tl_cuda_mem_info_get(args->src.info.buffer, data_len, team,
                                      &task->alltoall_ce.mem_info_src);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }
    status = ucc_tl_cuda_mem_info_get(args->dst.info.buffer, data_len, team,
                                      &task->alltoall_ce.mem_info_dst);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    task->super.post     = ucc_tl_cuda_alltoall_ce_start;
    task->super.progress = ucc_tl_cuda_alltoall_ce_progress;
    task->super.finalize = ucc_tl_cuda_alltoall_ce_finalize;
    task->bar            = TASK_BAR(task);

    return UCC_OK;

exit_err:
    ucc_ec_destroy_event(task->alltoall_ce.copy_done, UCC_EE_CUDA_STREAM);
    return status;
}
