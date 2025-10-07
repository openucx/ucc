/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatter/reduce_scatter.h"
#include <ucc/api/ucc.h>

#include "utils/arch/cuda_def.h"
#include "kernels/reduce_scatter_kernel.h"

#include <cuda_runtime.h>
#include <cuda.h>

enum {
    STAGE_COPY,          /*< Copy src buffer to symmetric memory */
    STAGE_COPY_BAR_START,
    STAGE_COPY_BAR_TEST,
    STAGE_KERNEL_START,
    STAGE_KERNEL,        /*< Kernel is running */
    STAGE_BARRIER_START, /*< Kernel is done, waiting for other ranks to finish */
    STAGE_BARRIER_TEST, /*< Kernel is done, waiting for other ranks to finish */
};

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task   = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_cuda_nvls_t *nvls   = &team->nvls;
    cudaStream_t        stream = team->stream;
    ucc_datatype_t      dt     = task->reduce_scatterv_nvls.dt;

    size_t src_size_bytes = args->src.info.count * ucc_dt_size(dt);
    size_t dst_size_bytes = args->dst.info.count * ucc_dt_size(dt);

    ucc_debug("reduce_scatterv_nvls_start symmetric uc addr: %lld mc addr: "
              "%lld src_size_bytes: %zu dst_size_bytes: %zu",
              nvls->uc_va, nvls->mc_va, src_size_bytes, dst_size_bytes);

    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        task->reduce_scatterv_nvls.rbuf = args->dst.info.buffer;
    } else {
        task->reduce_scatterv_nvls.rbuf = args->dst.info_v.buffer;
    }

    task->reduce_scatterv_nvls.sbuf = args->src.info.buffer;
    task->reduce_scatterv_nvls.src_size_bytes = src_size_bytes;
    task->reduce_scatterv_nvls.dst_size_bytes = dst_size_bytes;

    task->reduce_scatterv_nvls.stage = STAGE_COPY;

    // copy src buffer to symmetric memory first
    CUDA_CHECK(cudaMemcpyAsync((void *)nvls->uc_va, args->src.info.buffer,
                               src_size_bytes, cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaEventRecord(task->reduce_scatterv_nvls.evt_copy, stream));

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_reduce_scatterv_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task     = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    cudaEvent_t         evt      = task->reduce_scatterv_nvls.evt_completion;
    ucc_tl_cuda_nvls_t *nvls     = &team->nvls;
    cudaStream_t        stream   = team->stream;
    uint32_t            sm_count = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_sm_count;
    uint32_t            threads  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_threads;

    ucc_status_t        status;
    cudaError_t cuda_status;

    switch (task->reduce_scatterv_nvls.stage) {
    case STAGE_COPY:
        cuda_status = cudaEventQuery(task->reduce_scatterv_nvls.evt_copy);
        if (cuda_status == cudaErrorNotReady) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaEventQuery failed %s",
                      cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->reduce_scatterv_nvls.stage = STAGE_COPY_BAR_START;
        // fallthrough
    case STAGE_COPY_BAR_START:
        status = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
        if (status != UCC_OK) {
            ucc_error("reduce scatter barrier start failed");
            task->super.status = status;
            return;
        }
        task->reduce_scatterv_nvls.stage = STAGE_COPY_BAR_TEST;
        // fallthrough
    case STAGE_COPY_BAR_TEST:
        status = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->reduce_scatterv_nvls.stage = STAGE_KERNEL_START;
        // fallthrough
    case STAGE_KERNEL_START:
        status = post_reduce_scatter_kernel(stream, sm_count, threads, nvls->mc_va,
                                            (CUdeviceptr) task->reduce_scatterv_nvls.rbuf,
                                            task->reduce_scatterv_nvls.src_size_bytes, trank,
                                            UCC_TL_TEAM_SIZE(team));
        if (status != UCC_OK) {
            ucc_error("failed to post reduce scatter kernel");
            task->super.status = status;
            return;
        }
        ucc_debug("reduce scatter kernel posted");
        cuda_status = cudaEventRecord(task->reduce_scatterv_nvls.evt_completion, stream);
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaEventRecord failed: %s", cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->reduce_scatterv_nvls.stage = STAGE_KERNEL;
        // fallthrough
    case STAGE_KERNEL:
        cuda_status = cudaEventQuery(evt);
        if (cuda_status == cudaErrorNotReady) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaEventQuery failed %s",
                      cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->reduce_scatterv_nvls.stage = STAGE_BARRIER_START;
        // fallthrough
    case STAGE_BARRIER_START:
        status = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
        if (status != UCC_OK) {
            ucc_error("reduce scatter barrier start failed");
            task->super.status = status;
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
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);

    CUDA_CHECK(cudaEventDestroy(tl_task->reduce_scatterv_nvls.evt_completion));
    CUDA_CHECK(cudaEventDestroy(tl_task->reduce_scatterv_nvls.evt_copy));

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

ucc_status_t
ucc_tl_cuda_reduce_scatter_nvls_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t      *tl_team,
                                     ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (coll_args->args.op != UCC_OP_SUM ||
        coll_args->args.dst.info.datatype != UCC_DT_FLOAT32) {
        ucc_error("NVLS reduce scatter is supported only with SUM operation "
                  "and float32 datatype");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo))) {
        ucc_error("NVLS reduce scatter is supported only on fully connected "
                  "NVLINK systems");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    CUDA_CHECK(cudaEventCreateWithFlags(
        &task->reduce_scatterv_nvls.evt_completion, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &task->reduce_scatterv_nvls.evt_copy, cudaEventDisableTiming));

    task->reduce_scatterv_nvls.get_count  = ucc_tl_cuda_reduce_scatter_get_count;
    task->reduce_scatterv_nvls.get_offset = ucc_tl_cuda_reduce_scatter_get_offset;
    task->reduce_scatterv_nvls.dt         = coll_args->args.dst.info.datatype;

    task->super.post     = ucc_tl_cuda_reduce_scatterv_nvls_start;
    task->super.progress = ucc_tl_cuda_reduce_scatterv_nvls_progress;
    task->super.finalize = ucc_tl_cuda_reduce_scatterv_nvls_finalize;

    task->bar = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
