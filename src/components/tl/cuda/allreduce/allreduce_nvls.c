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
    STAGE_KERNEL, /*< Post memcpy to symmetric buffer, launch kernel, memcpy to destination */
    STAGE_WAIT, /*< Wait for the copies and kernel to complete */
};

ucc_status_t ucc_tl_cuda_allreduce_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_ee_h            ee   = task->super.ee;
    cudaStream_t stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    // stream is used in tl_trace below, but not used in this function
    (void)stream;

    task->allreduce_nvls.buf_size_bytes = args->dst.info.count *
                                          ucc_dt_size(task->allreduce_nvls.dt);
    task->allreduce_nvls.rbuf = args->dst.info.buffer;
    task->allreduce_nvls.sbuf = UCC_IS_INPLACE(*args) ? args->dst.info.buffer
                                                      : args->src.info.buffer;

    tl_trace(
        UCC_TASK_LIB(task),
        "task: %p stream: %p allreduce_nvls_start symmetric uc addr: %p "
        "mc addr: %p "
        "buf_size_bytes: %zu, is inplace: %d",
        task,
        stream,
        (void *)task->allreduce_nvls.uc_va,
        (void *)task->allreduce_nvls.mc_va,
        task->allreduce_nvls.buf_size_bytes,
        UCC_IS_INPLACE(*args));

    task->allreduce_nvls.stage = STAGE_KERNEL;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_allreduce_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t  *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t  *team  = TASK_TEAM(task);
    ucc_rank_t           trank = UCC_TL_TEAM_RANK(team);
    ucc_ec_cuda_event_t *ec_event = (ucc_ec_cuda_event_t *)
                                        task->allreduce_nvls.evt_completion;
    cudaEvent_t    evt    = ec_event->event;
    CUdeviceptr    mc_va  = task->allreduce_nvls.mc_va;
    CUdeviceptr    uc_va  = task->allreduce_nvls.uc_va;
    ucc_ee_h       ee     = task->super.ee;
    cudaStream_t   stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    ucc_datatype_t dt     = task->allreduce_nvls.dt;
    uint32_t       sm_count = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_sm_count;
    uint32_t       threads  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_threads;

    ucc_status_t   status;
    cudaError_t    cuda_status;

    switch (task->allreduce_nvls.stage) {
    case STAGE_KERNEL:
        // copy src buffer to symmetric memory first
        status = CUDA_FUNC(cudaMemcpyAsync(
            (void *)uc_va,
            task->allreduce_nvls.sbuf,
            task->allreduce_nvls.buf_size_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }

        status = post_allreduce_kernel(
            stream,
            sm_count,
            threads,
            mc_va,
            task->allreduce_nvls.buf_size_bytes,
            TASK_NVLS_CONTROL_MC(task),
            TASK_NVLS_CONTROL_UC(task),
            task->allreduce_nvls.coll_id,
            trank,
            UCC_TL_TEAM_SIZE(team),
            dt);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to post allreduce kernel");
            task->super.status = status;
            return;
        }
        status = CUDA_FUNC(cudaMemcpyAsync(
            (void *)task->allreduce_nvls.rbuf,
            (void *)uc_va,
            task->allreduce_nvls.buf_size_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        status = CUDA_FUNC(cudaEventRecord(evt, stream));
        if (status != UCC_OK) {
            task->super.status = status;
            return;
        }
        task->allreduce_nvls.stage = STAGE_WAIT;
        // fallthrough
    case STAGE_WAIT:
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

    ucc_ec_destroy_event(
        tl_task->allreduce_nvls.evt_completion, UCC_EE_CUDA_STREAM);

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

// NOLINTNEXTLINE(misc-unused-parameters): ev parameter unused as it's not needed for this implementation
ucc_status_t ucc_tl_cuda_allreduce_nvls_triggered_post(
    ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_status_t        status;
    ucc_ev_t            post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_trace(UCC_TASK_LIB(task), "triggered post. task:%p", coll_task);

    task->allreduce_nvls.stage = STAGE_KERNEL;

    status                     = coll_task->post(coll_task);
    if (ucc_likely(status == UCC_OK)) {
        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.ev_context      = NULL;
        post_event.req             = &coll_task->super;
        status                     = ucc_ee_set_event_internal(
            coll_task->ee, &post_event, &coll_task->ee->event_out_queue);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(
                UCC_TASK_LIB(task),
                "failed to set EE event: %s",
                ucc_status_string(status));
            return status;
        }
    }
    return status;
}

ucc_status_t ucc_tl_cuda_allreduce_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team     = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    size_t              buf_size = coll_args->args.dst.info.count *
                      ucc_dt_size(coll_args->args.dst.info.datatype);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (buf_size < 1024 || coll_args->args.op != UCC_OP_SUM ||
        (coll_args->args.dst.info.datatype != UCC_DT_FLOAT32 &&
         coll_args->args.dst.info.datatype != UCC_DT_BFLOAT16)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS allreduce is supported only with SUM operation "
            "and float32 or bfloat16 datatype, with message size >= 1024 "
            "bytes");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to initialize CUDA task");
        return status;
    }

    status = ucc_ec_create_event(
        &task->allreduce_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create CUDA event");
        return status;
    }

    task->allreduce_nvls.dt = coll_args->args.dst.info.datatype;
    tl_trace(
        UCC_TL_TEAM_LIB(team),
        "NVLS allreduce datatype: %s",
        ucc_datatype_str(task->allreduce_nvls.dt));

    task->super.post             = ucc_tl_cuda_allreduce_nvls_start;
    task->super.triggered_post   = ucc_tl_cuda_allreduce_nvls_triggered_post;
    task->super.progress         = ucc_tl_cuda_allreduce_nvls_progress;
    task->super.finalize         = ucc_tl_cuda_allreduce_nvls_finalize;

    task->bar                    = TASK_BAR(task);

    task->allreduce_nvls.uc_va   = (CUdeviceptr)TASK_SYMMETRIC_UC(task);
    task->allreduce_nvls.mc_va   = (CUdeviceptr)TASK_SYMMETRIC_MC(task);

    task->allreduce_nvls.coll_id = team->nvls.coll_ids[task->coll_id]++;

    *task_p                      = &task->super;
    return UCC_OK;
}
