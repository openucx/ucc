/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

// #include "reduce_scatterv/reduce_scatterv.h"
#include "reduce_scatter/reduce_scatter.h"
#include <ucc/api/ucc.h>

#include "utils/arch/cuda_def.h"

#include <cuda_runtime.h>
#include <cuda.h>

enum
{
    STAGE_KERNEL,  /*< Kernel is running */
    STAGE_BARRIER_START, /*< Kernel is done, waiting for other ranks to finish */
    STAGE_BARRIER_TEST, /*< Kernel is done, waiting for other ranks to finish */
};

ucc_status_t
post_reduce_scatter_kernel(cudaStream_t stream, CUdeviceptr src_addr, CUdeviceptr dst_addr, size_t data_size, uint32_t rank, uint32_t tsize);


ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *   args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt    = task->reduce_scatterv_nvls.dt;
    cudaStream_t        stream = team->stream;

    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        task->reduce_scatterv_nvls.rbuf = args->dst.info.buffer;
    } else {
        task->reduce_scatterv_nvls.rbuf = args->dst.info_v.buffer;
    }

    task->reduce_scatterv_nvls.sbuf  = args->src.info.buffer;

    task->reduce_scatterv_nvls.stage = STAGE_KERNEL;
    
    // if (task->reduce_scatterv_nvls.evtCompletion != NULL) {
    //     CUDA_CHECK(cudaEventDestroy(task->reduce_scatterv_nvls.evtCompletion));
    // }

    CUDA_CHECK(cudaEventCreateWithFlags(&task->reduce_scatterv_nvls.evtCompletion, cudaEventDisableTiming));

    size_t src_size_bytes = args->src.info.count * ucc_dt_size(dt);
    size_t dst_size_bytes = args->dst.info.count * ucc_dt_size(dt);

    ucc_debug("reduce_scatterv_nvls_start symmetric uc addr: %lld mc addr: %lld src_size_bytes: %zu dst_size_bytes: %zu", team->uc_va, team->mc_va, src_size_bytes, dst_size_bytes);

    CUDA_CHECK(cudaMemcpyAsync((void *)team->uc_va, args->src.info.buffer, src_size_bytes,
                    cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ucc_status_t status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (status != UCC_OK) {
        ucc_error("reduce scatter barrier start failed");
        return UCC_ERR_NO_RESOURCE;
    }

    while (ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar) == UCC_INPROGRESS) {
    }

    ucc_debug("memcpy done on all ranks");

    // memcpy done on all ranks so we can start the kernel

    status = post_reduce_scatter_kernel(
        stream, team->mc_va, (CUdeviceptr)args->dst.info.buffer, src_size_bytes,
        UCC_TL_TEAM_RANK(team), tsize);
    if (status != UCC_OK) {
        ucc_error("failed to post reduce scatter kernel");
        return status;
    }
    ucc_debug("reduce scatter kernel posted");

    CUDA_CHECK(cudaEventRecord(task->reduce_scatterv_nvls.evtCompletion, stream));

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_reduce_scatterv_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_rank_t trank = UCC_TL_TEAM_RANK(team);
    cudaEvent_t evt = task->reduce_scatterv_nvls.evtCompletion;
    ucc_status_t status;

    switch (task->reduce_scatterv_nvls.stage) {
        case STAGE_KERNEL:
            cudaError_t cuda_status = cudaEventQuery(evt);
            if (cuda_status == cudaErrorNotReady) {
                // ucc_print("reduce scatter kernel is not completed yet");
                task->super.status = UCC_INPROGRESS;
                return;
            }
            if (cuda_status != cudaSuccess) {
                ucc_error("cudaEventQuery failed %s", cudaGetErrorString(cuda_status));
                task->super.status = UCC_ERR_NO_RESOURCE;
                return;
            }
            cuda_status = cudaEventDestroy(evt);
            if (cuda_status != cudaSuccess) {
                ucc_error("cudaEventDestroy failed");
                task->super.status = UCC_ERR_NO_RESOURCE;
                return;
            }
            task->reduce_scatterv_nvls.stage = STAGE_BARRIER_START;
            // fallthrough
        case STAGE_BARRIER_START:
            status = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
            if (status != UCC_OK) {
                ucc_error("reduce scatter barrier start failed");
                task->super.status = UCC_ERR_NO_RESOURCE;
                return;
            }
            task->reduce_scatterv_nvls.stage = STAGE_BARRIER_TEST;
            // fallthrough
        case STAGE_BARRIER_TEST:
            status = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
            if (status != UCC_OK) {
                task->super.status = status;
                return;
            }
            task->super.status = UCC_OK;
            ucc_debug("reduce scatter kernel is completed");
            break;
    }
    return;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_nvls_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *     tl_team,
                                                  ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
        UCC_TL_TEAM_SIZE(team) - 1 > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->reduce_scatterv_nvls.get_count  = ucc_tl_cuda_reduce_scatter_get_count;
    task->reduce_scatterv_nvls.get_offset = ucc_tl_cuda_reduce_scatter_get_offset;
    task->reduce_scatterv_nvls.dt         = coll_args->args.dst.info.datatype;
    // task->super.flags          |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_reduce_scatterv_nvls_start;
    task->super.progress       = ucc_tl_cuda_reduce_scatterv_nvls_progress;
    task->super.finalize       = ucc_tl_cuda_reduce_scatterv_nvls_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
