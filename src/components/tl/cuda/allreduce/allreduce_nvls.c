/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce/allreduce.h"
#include "ucc/api/ucc.h"
#include "utils/arch/cuda_def.h"

#include <cuda_runtime.h>
#include <cuda.h>

enum {
    STAGE_COPY, /*< Copy src buffer to symmetric memory */
    STAGE_COPY_BAR_START,
    STAGE_COPY_BAR_TEST,
    STAGE_KERNEL_START,
    STAGE_KERNEL,        /*< Kernel is running */
    STAGE_BARRIER_START, /*< Kernel is done, waiting for other ranks to finish */
    STAGE_BARRIER_TEST,  /*< Kernel is done, waiting for other ranks to finish */
    STAGE_COPY_POST,
    STAGE_COPY_POST_WAIT,
    STAGE_COPY_POST_BAR_START,
    STAGE_COPY_POST_BAR_TEST
};

// Kernel is defined in src/components/tl/cuda/kernels/allreduce_kernel.cu
ucc_status_t post_allreduce_kernel(cudaStream_t stream, CUdeviceptr src_addr,
                                   size_t src_size_bytes, uint32_t rank,
                                   uint32_t tsize);

ucc_status_t ucc_tl_cuda_allreduce_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task   = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_cuda_nvls_t *nvls   = &team->nvls;
    cudaStream_t        stream = team->stream;
    ucc_datatype_t      dt     = task->allreduce_nvls.dt;

    size_t src_size_bytes = args->src.info.count * ucc_dt_size(dt);
    size_t dst_size_bytes = args->dst.info.count * ucc_dt_size(dt);

    ucc_debug("allreduce_nvls_start symmetric uc addr: %lld mc addr: "
              "%lld src_size_bytes: %zu dst_size_bytes: %zu",
              nvls->uc_va, nvls->mc_va, src_size_bytes, dst_size_bytes);

    task->allreduce_nvls.rbuf           = args->dst.info.buffer;
    task->allreduce_nvls.sbuf           = args->src.info.buffer;
    task->allreduce_nvls.src_size_bytes = src_size_bytes;
    task->allreduce_nvls.dst_size_bytes = dst_size_bytes;

    task->allreduce_nvls.stage = STAGE_COPY;

    // copy src buffer to symmetric memory first
    CUDA_CHECK(cudaMemcpyAsync((void *)nvls->uc_va, args->src.info.buffer,
                               src_size_bytes, cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaEventRecord(task->allreduce_nvls.evtCompletion, stream));

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_allreduce_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task   = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          trank  = UCC_TL_TEAM_RANK(team);
    cudaEvent_t         evt    = task->allreduce_nvls.evtCompletion;
    ucc_tl_cuda_nvls_t *nvls   = &team->nvls;
    cudaStream_t        stream = team->stream;

    ucc_status_t        status;
    cudaError_t         cuda_status;
    switch (task->allreduce_nvls.stage) {
    case STAGE_COPY:
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
        task->allreduce_nvls.stage = STAGE_COPY_BAR_START;
        // fallthrough
    case STAGE_COPY_BAR_START:
        status = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
        if (status != UCC_OK) {
            ucc_error("reduce scatter barrier start failed");
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->allreduce_nvls.stage = STAGE_COPY_BAR_TEST;
        // fallthrough
    case STAGE_COPY_BAR_TEST:
        status = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->allreduce_nvls.stage = STAGE_KERNEL_START;
        // fallthrough
    case STAGE_KERNEL_START:
        status = post_allreduce_kernel(stream, nvls->mc_va,
                                       task->allreduce_nvls.src_size_bytes,
                                       trank, UCC_TL_TEAM_SIZE(team));
        if (status != UCC_OK) {
            ucc_error("failed to post allreduce kernel");
            task->super.status = status;
            return;
        }
        ucc_debug("allreduce kernel posted");
        cuda_status = cudaEventRecord(evt, stream);
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaEventRecord failed: %s", cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->allreduce_nvls.stage = STAGE_KERNEL;
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
        task->allreduce_nvls.stage = STAGE_BARRIER_START;
        // fallthrough
    case STAGE_BARRIER_START:
        status = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
        if (status != UCC_OK) {
            ucc_error("reduce scatter barrier start failed");
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->allreduce_nvls.stage = STAGE_BARRIER_TEST;
        // fallthrough
    case STAGE_BARRIER_TEST:
        status = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        ucc_debug("allreduce kernel is completed");
        task->allreduce_nvls.stage = STAGE_COPY_POST;
        // fallthrough        
    case STAGE_COPY_POST:
        cudaMemcpyAsync((void *)task->allreduce_nvls.rbuf,
                        (void *)nvls->uc_va,
                        task->allreduce_nvls.dst_size_bytes,
                        cudaMemcpyDeviceToDevice,
                        stream);
        cuda_status = cudaEventRecord(evt, stream);
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaEventRecord failed: %s", cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->allreduce_nvls.stage = STAGE_COPY_POST_WAIT;
        // fallthrough
    case STAGE_COPY_POST_WAIT:
        cuda_status = cudaEventQuery(evt);
        if (cuda_status == cudaErrorNotReady) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        task->allreduce_nvls.stage = STAGE_COPY_BAR_START;
        // fallthrough
    case STAGE_COPY_POST_BAR_START:
        status = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
        if (status != UCC_OK) {
            ucc_error("reduce scatter barrier start failed");
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->allreduce_nvls.stage = STAGE_COPY_POST_BAR_TEST;
        // fallthrough
    case STAGE_COPY_POST_BAR_TEST:
        status = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->super.status = UCC_OK;
        ucc_debug("allreduce kernel is completed");
        break;
    }
    return;
}

ucc_status_t ucc_tl_cuda_allreduce_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);

    CUDA_CHECK(cudaEventDestroy(tl_task->allreduce_nvls.evtCompletion));

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allreduce_nvls_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *tl_team,
                                             ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (coll_args->args.op != UCC_OP_SUM ||
        coll_args->args.dst.info.datatype != UCC_DT_FLOAT32) {
        ucc_error("NVLS allreduce is supported only with SUM operation "
                  "and float32 datatype");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo))) {
        ucc_error("NVLS allreduce is supported only on fully connected "
                  "NVLINK systems");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    CUDA_CHECK(cudaEventCreateWithFlags(&task->allreduce_nvls.evtCompletion,
                                        cudaEventDisableTiming));

    task->allreduce_nvls.dt = coll_args->args.dst.info.datatype;

    task->super.post     = ucc_tl_cuda_allreduce_nvls_start;
    task->super.progress = ucc_tl_cuda_allreduce_nvls_progress;
    task->super.finalize = ucc_tl_cuda_allreduce_nvls_finalize;

    task->bar = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
