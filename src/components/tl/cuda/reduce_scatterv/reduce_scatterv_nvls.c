/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv/reduce_scatterv.h"
#include <ucc/api/ucc.h>

#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"
#include "tl_cuda_nvls.h"
#include "kernels/reduce_scatter_kernel.h"

#include "components/ec/ucc_ec.h"
#include "components/ec/cuda/ec_cuda_resources.h"

enum {
    STAGE_KERNEL, /*< Post memcpy to symmetric buffer, launch kernel, memcpy to destination */
    STAGE_WAIT, /*< Wait for the copies and kernel to complete */
};

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_datatype_t      dt   = task->reduce_scatterv_nvls.dt;

    task->reduce_scatterv_nvls.src_size_bytes = args->src.info.count *
                                                ucc_dt_size(dt);

    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTER) {
        task->reduce_scatterv_nvls.rbuf = args->dst.info.buffer;
        task->reduce_scatterv_nvls.sbuf = UCC_IS_INPLACE(*args)
                                              ? args->dst.info.buffer
                                              : args->src.info.buffer;
    } else {
        task->reduce_scatterv_nvls.rbuf = args->dst.info_v.buffer;
        task->reduce_scatterv_nvls.sbuf = UCC_IS_INPLACE(*args)
                                              ? args->dst.info_v.buffer
                                              : args->src.info_v.buffer;
    }

    task->reduce_scatterv_nvls.stage = STAGE_KERNEL;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_reduce_scatterv_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t  *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t  *team = TASK_TEAM(task);
    ucc_ec_cuda_event_t *ec_event = (ucc_ec_cuda_event_t *)task
                                        ->reduce_scatterv_nvls.evt_completion;
    cudaEvent_t    evt    = ec_event->event;
    CUdeviceptr    mc_va  = task->reduce_scatterv_nvls.mc_va;
    CUdeviceptr    uc_va  = task->reduce_scatterv_nvls.uc_va;
    ucc_ee_h       ee     = task->super.ee;
    cudaStream_t   stream = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    ucc_datatype_t dt     = task->reduce_scatterv_nvls.dt;
    uint32_t       sm_count = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_sm_count;
    uint32_t       threads  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_threads;
    ucc_status_t   status;
    cudaError_t    cuda_status;

    switch (task->reduce_scatterv_nvls.stage) {
    case STAGE_KERNEL:
        /* copy src buffer to symmetric memory first */
        status = CUDA_FUNC(cudaMemcpyAsync(
            (void *)uc_va,
            task->reduce_scatterv_nvls.sbuf,
            task->reduce_scatterv_nvls.src_size_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }

        status = post_reduce_scatter_kernel(
            stream,
            sm_count,
            threads,
            (CUdeviceptr)task->reduce_scatterv_nvls.rbuf,
            mc_va,
            task->reduce_scatterv_nvls.src_size_bytes,
            TASK_NVLS_CONTROL_MC(task),
            TASK_NVLS_CONTROL_UC(task),
            task->reduce_scatterv_nvls.coll_id,
            task->reduce_scatterv_nvls.offset,
            task->reduce_scatterv_nvls.count,
            dt,
            UCC_TL_TEAM_SIZE(team));
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(
                UCC_TASK_LIB(task), "failed to post reduce scatter kernel");
            task->super.status = status;
            return;
        }
        status = CUDA_FUNC(cudaEventRecord(evt, stream));
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->reduce_scatterv_nvls.stage = STAGE_WAIT;
        /* fallthrough */
    case STAGE_WAIT:
        cuda_status = cudaEventQuery(evt);
        if (cuda_status == cudaErrorNotReady) {
            task->super.status = UCC_INPROGRESS;
            return;
        } else if (ucc_unlikely(cuda_status != cudaSuccess)) {
            tl_error(
                UCC_TASK_LIB(task),
                "error cudaEventQuery %s!",
                cudaGetErrorString(cuda_status));
            task->super.status = UCC_ERR_NO_MESSAGE;
            return;
        }
        task->super.status = UCC_OK;
        break;
    }
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_triggered_post(
    ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task) // NOLINT
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_status_t        status;
    ucc_ev_t            post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_trace(UCC_TASK_LIB(task), "triggered post. task:%p", coll_task);

    status = coll_task->post(coll_task);
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

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);
    tl_trace(
        UCC_TASK_LIB(tl_task), "task: %p reduce_scatterv_nvls_finalize", task);

    ucc_ec_destroy_event(
        tl_task->reduce_scatterv_nvls.evt_completion, UCC_EE_CUDA_STREAM);

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team  = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_datatype_t      dt    = coll_args->args.dst.info_v.datatype;
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;
    size_t              offset_elements;
    size_t              count_elements;

    if (coll_args->args.op != UCC_OP_SUM) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS reduce scatter v is supported only with SUM operation");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (dt != UCC_DT_FLOAT32 && dt != UCC_DT_BFLOAT16) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS reduce scatter v is supported only with float32 or bfloat16 "
            "datatype");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to initialize CUDA task");
        return status;
    }

    status = ucc_ec_create_event(
        &task->reduce_scatterv_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create CUDA event");
        ucc_tl_cuda_task_put(task);
        return status;
    }

    task->reduce_scatterv_nvls.dt = dt;

    /* Get offset and count in datatype elements, then convert to uint32_t units.
     * Use aligned offset for NVLS to satisfy 16-byte alignment requirement
     * of multimem instructions (must be multiple of 4 uint32_t = 16 bytes). */
    offset_elements = ucc_tl_cuda_reduce_scatterv_get_offset(task, trank);
    count_elements  = ucc_tl_cuda_reduce_scatterv_get_count(task, trank);

    /* Convert from datatype elements to uint32_t indices for the kernel.
     * For float32: 1 element = 1 uint32_t
     * For bfloat16: 2 elements = 1 uint32_t */
    if (dt == UCC_DT_FLOAT32) {
        task->reduce_scatterv_nvls.offset = offset_elements;
        task->reduce_scatterv_nvls.count  = count_elements;
    } else { /* UCC_DT_BFLOAT16 */
        if (offset_elements % 2 != 0 || count_elements % 2 != 0) {
            tl_debug(
                UCC_TL_TEAM_LIB(team),
                "BF16 offset and count must be even, got offset=%zu count=%zu",
                offset_elements,
                count_elements);
            goto err_cleanup;
        }
        task->reduce_scatterv_nvls.offset = offset_elements / 2;
        task->reduce_scatterv_nvls.count  = count_elements / 2;
    }

    /* NVLS requires 16-byte alignment (4 uint32_t elements) */
    if (ucc_unlikely(task->reduce_scatterv_nvls.offset % 4 != 0)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS requires 16-byte alignment for offset, got offset=%zu "
            "(uint32_t units)",
            task->reduce_scatterv_nvls.offset);
        goto err_cleanup;
    }
    if (ucc_unlikely(task->reduce_scatterv_nvls.count % 4 != 0)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS requires 16-byte alignment for count, got count=%zu "
            "(uint32_t units)",
            task->reduce_scatterv_nvls.count);
        goto err_cleanup;
    }

    task->reduce_scatterv_nvls.mc_va   = (CUdeviceptr)TASK_SYMMETRIC_MC(task);
    task->reduce_scatterv_nvls.uc_va   = (CUdeviceptr)TASK_SYMMETRIC_UC(task);
    task->reduce_scatterv_nvls.coll_id = team->nvls.coll_ids[task->coll_id]++;

    task->super.post                   = ucc_tl_cuda_reduce_scatterv_nvls_start;
    task->super
        .triggered_post  = ucc_tl_cuda_reduce_scatterv_nvls_triggered_post;
    task->super.progress = ucc_tl_cuda_reduce_scatterv_nvls_progress;
    task->super.finalize = ucc_tl_cuda_reduce_scatterv_nvls_finalize;

    *task_p              = &task->super;
    return UCC_OK;

err_cleanup:
    ucc_ec_destroy_event(
        task->reduce_scatterv_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    ucc_tl_cuda_task_put(task);
    return UCC_ERR_NOT_SUPPORTED;
}
