/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce/allreduce.h"
#include "ucc/api/ucc.h"
#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"
#include "tl_cuda_nvls.h"
#include "../kernels/allreduce_kernel.h"
#include "components/ec/ucc_ec.h"
#include "components/ec/cuda/ec_cuda_resources.h"


enum {
    STAGE_COPY,                /*< Copy src buffer to symmetric memory */
    STAGE_COPY_WAIT,           /*< Wait for the copy to complete */
    STAGE_COPY_BAR_START,      /*< Start barrier after copy */
    STAGE_COPY_BAR_TEST,       /*< Test barrier after copy */
    STAGE_KERNEL_START,        /*< Start kernel */
    STAGE_KERNEL,              /*< Kernel is running */
    STAGE_BARRIER_START,       /*< Start barrier after kernel */
    STAGE_BARRIER_TEST,        /*< Test barrier after kernel */
    STAGE_COPY_POST,           /*< Copy result buffer from symmetric memory to dst buffer */
    STAGE_COPY_POST_WAIT,      /*< Wait for the copy to complete */
};

ucc_status_t ucc_tl_cuda_allreduce_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task   = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_ee_h            ee     = task->super.ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    // stream is used in tl_trace below, but not used in this function
    (void)stream;

    task->allreduce_nvls.buf_size_bytes = args->dst.info.count * ucc_dt_size(task->allreduce_nvls.dt);
    task->allreduce_nvls.rbuf = args->dst.info.buffer;
    task->allreduce_nvls.sbuf =
        UCC_IS_INPLACE(*args) ? args->dst.info.buffer : args->src.info.buffer;

    tl_trace(UCC_TASK_LIB(task),
             "task: %p stream: %p allreduce_nvls_start symmetric uc addr: %p "
             "mc addr: %p "
             "buf_size_bytes: %zu, is inplace: %d",
             task, stream, (void *)task->allreduce_nvls.uc_va, (void *)task->allreduce_nvls.mc_va,
             task->allreduce_nvls.buf_size_bytes, UCC_IS_INPLACE(*args));

    task->allreduce_nvls.stage = STAGE_COPY;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_allreduce_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task      = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team      = TASK_TEAM(task);
    ucc_rank_t          trank     = UCC_TL_TEAM_RANK(team);
    ucc_ec_cuda_event_t *ec_event = (ucc_ec_cuda_event_t *)task->allreduce_nvls.evtCompletion;
    cudaEvent_t         evt       = ec_event->event;
    CUdeviceptr         mc_va     = task->allreduce_nvls.mc_va;
    CUdeviceptr         uc_va     = task->allreduce_nvls.uc_va;
    ucc_ee_h            ee        = task->super.ee;
    cudaStream_t        stream    = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    ucc_datatype_t      dt        = task->allreduce_nvls.dt;
    uint32_t            sm_count  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_sm_count;
    uint32_t            threads   = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_threads;

    ucc_status_t        status;
    cudaError_t         cuda_status;

    switch (task->allreduce_nvls.stage) {
    case STAGE_COPY:
        // copy src buffer to symmetric memory first
        cuda_status =
            cudaMemcpyAsync((void *)uc_va, task->allreduce_nvls.sbuf,
                            task->allreduce_nvls.buf_size_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaMemcpyAsync failed: %s",
                      cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_MEMORY; // TODO: better error code?
            return;
        }
        cuda_status = cudaEventRecord(
            ((ucc_ec_cuda_event_t *)task->allreduce_nvls.evtCompletion)->event,
            stream);
        if (cuda_status != cudaSuccess) {
            ucc_error("cudaEventRecord failed: %s",
                      cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
        task->allreduce_nvls.stage = STAGE_COPY_WAIT;
        // fallthrough
    case STAGE_COPY_WAIT:
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
            ucc_error("allreduce barrier start failed");
            task->super.status = status;
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
        status = post_allreduce_kernel(stream, sm_count, threads, mc_va,
                                       task->allreduce_nvls.buf_size_bytes,
                                       trank, UCC_TL_TEAM_SIZE(team), dt);
        if (status != UCC_OK) {
            ucc_error("failed to post allreduce kernel");
            task->super.status = status;
            return;
        }
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
            ucc_error("allreduce barrier start failed");
            task->super.status = status;
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
        tl_trace(UCC_TASK_LIB(task), "task: %p allreduce kernel is completed", task);
        task->allreduce_nvls.stage = STAGE_COPY_POST;
        // fallthrough
    case STAGE_COPY_POST:
        cuda_status = cudaMemcpyAsync((void *)task->allreduce_nvls.rbuf,
                        (void *)uc_va,
                        task->allreduce_nvls.buf_size_bytes,
                        cudaMemcpyDeviceToDevice,
                        stream);
        if (cuda_status != cudaSuccess) {
            ucc_error("task: %p, cudaMemcpyAsync failed: %s, stream: %p, sbuf: %p, rbuf: %p, uc_va: %p, buf_size_bytes: %zu", task, cudaGetErrorString(cuda_status), stream, task->allreduce_nvls.sbuf, task->allreduce_nvls.rbuf, (void*) uc_va, task->allreduce_nvls.buf_size_bytes);
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
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
        task->super.status = UCC_OK;
        break;
    }
}

ucc_status_t ucc_tl_cuda_allreduce_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(tl_task), "task: %p allreduce_nvls_finalize", task);

    ucc_ec_destroy_event(tl_task->allreduce_nvls.evtCompletion, UCC_EE_CUDA_STREAM);

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

//NOLINTNEXTLINE(misc-unused-parameters): ev parameter unused as it's not needed for this implementation
ucc_status_t ucc_tl_cuda_allreduce_nvls_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                                     ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_status_t        status;
    ucc_ev_t            post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_debug(UCC_TASK_LIB(task), "triggered post. task:%p", coll_task);

    task->allreduce_nvls.stage = STAGE_COPY;

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

ucc_status_t ucc_tl_cuda_allreduce_nvls_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *tl_team,
                                             ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (coll_args->args.op != UCC_OP_SUM ||
        (coll_args->args.dst.info.datatype != UCC_DT_FLOAT32 &&
         coll_args->args.dst.info.datatype != UCC_DT_BFLOAT16)) {
        ucc_error("NVLS allreduce is supported only with SUM operation "
                  "and float32 or bfloat16 datatype");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo))) {
        ucc_error("NVLS allreduce is supported only on fully connected "
                  "NVLINK systems");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to initialize CUDA task");
        return status;
    }

    status = ucc_ec_create_event(&task->allreduce_nvls.evtCompletion, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to create CUDA event");
        ucc_tl_cuda_task_put(task);
        return status;
    }

    task->allreduce_nvls.dt = coll_args->args.dst.info.datatype;
    ucc_debug("NVLS allreduce datatype: %ld", (long)task->allreduce_nvls.dt);

    task->super.post     = ucc_tl_cuda_allreduce_nvls_start;
    task->super.triggered_post = ucc_tl_cuda_allreduce_nvls_triggered_post;
    task->super.progress = ucc_tl_cuda_allreduce_nvls_progress;
    task->super.finalize = ucc_tl_cuda_allreduce_nvls_finalize;

    task->bar = TASK_BAR(task);

    task->allreduce_nvls.uc_va = (CUdeviceptr) TASK_SYMMETRIC_UC(task);
    task->allreduce_nvls.mc_va = (CUdeviceptr) TASK_SYMMETRIC_MC(task);

    *task_p = &task->super;
    return UCC_OK;
}
