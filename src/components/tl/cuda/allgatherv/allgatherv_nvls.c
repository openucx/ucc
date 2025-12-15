/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv/allgatherv.h"
#include <ucc/api/ucc.h>

#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"
#include "tl_cuda_nvls.h"
#include "kernels/allgatherv_kernel.h"

#include "components/ec/ucc_ec.h"
#include "components/ec/cuda/ec_cuda_resources.h"

enum {
    STAGE_KERNEL, /*< Post memcpy to symmetric buffer, wait for completion */
    STAGE_WAIT,   /*< Wait for the kernel to complete */
};

ucc_status_t ucc_tl_cuda_allgatherv_nvls_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_rank_t          trank   = UCC_TL_TEAM_RANK(team);
    ucc_datatype_t      dt      = task->allgatherv_nvls.dt;
    size_t              dt_size = ucc_dt_size(dt);

    if (args->coll_type == UCC_COLL_TYPE_ALLGATHERV) {
        task->allgatherv_nvls.rbuf = args->dst.info_v.buffer;
        task->allgatherv_nvls
            .sbuf = UCC_IS_INPLACE(*args)
                        ? PTR_OFFSET(
                              args->dst.info_v.buffer,
                              ucc_tl_cuda_allgatherv_get_offset(task, trank) *
                                  dt_size)
                        : args->src.info.buffer;
    } else {
        task->allgatherv_nvls.rbuf = args->dst.info.buffer;
        task->allgatherv_nvls
            .sbuf = UCC_IS_INPLACE(*args)
                        ? PTR_OFFSET(
                              args->dst.info.buffer,
                              ucc_tl_cuda_allgatherv_get_offset(task, trank) *
                                  dt_size)
                        : args->src.info.buffer;
    }

    task->allgatherv_nvls.stage = STAGE_KERNEL;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_cuda_allgatherv_nvls_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t  *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t  *team = TASK_TEAM(task);
    ucc_ec_cuda_event_t *ec_event = (ucc_ec_cuda_event_t *)
                                        task->allgatherv_nvls.evt_completion;
    cudaEvent_t  evt      = ec_event->event;
    CUdeviceptr  mc_va    = task->allgatherv_nvls.mc_va;
    CUdeviceptr  uc_va    = task->allgatherv_nvls.uc_va;
    ucc_ee_h     ee       = task->super.ee;
    cudaStream_t stream   = (ee) ? (cudaStream_t)ee->ee_context : team->stream;
    uint32_t     sm_count = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_sm_count;
    uint32_t     threads  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_threads;
    ucc_status_t status;
    cudaError_t  cuda_status;

    switch (task->allgatherv_nvls.stage) {
    case STAGE_KERNEL:
        /* Each rank copies its data to the NVLS buffer at its specific offset
         * using CUDA memcpy to mc ptr */
        status = post_allgatherv_kernel(
            stream,
            sm_count,
            threads,
            (CUdeviceptr)task->allgatherv_nvls.sbuf,
            mc_va,
            task->allgatherv_nvls.offset,
            task->allgatherv_nvls.count,
            TASK_NVLS_CONTROL_MC(task),
            TASK_NVLS_CONTROL_UC(task),
            task->allgatherv_nvls.coll_id,
            UCC_TL_TEAM_SIZE(team));
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task), "failed to post allgatherv kernel");
            task->super.status = status;
            return;
        }

        /* Copy gathered data from uc ptr (batched wait) to destination buffer
         * total_count is in uint32_t units, convert to bytes */
        status = CUDA_FUNC(cudaMemcpyAsync(
            task->allgatherv_nvls.rbuf,
            (void *)uc_va,
            task->allgatherv_nvls.total_count * sizeof(uint32_t),
            cudaMemcpyDeviceToDevice,
            stream));
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }

        status = CUDA_FUNC(cudaEventRecord(evt, stream));
        if (ucc_unlikely(status != UCC_OK)) {
            task->super.status = status;
            return;
        }
        task->allgatherv_nvls.stage = STAGE_WAIT;
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

ucc_status_t ucc_tl_cuda_allgatherv_nvls_triggered_post(
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

ucc_status_t ucc_tl_cuda_allgatherv_nvls_finalize(ucc_coll_task_t *task)
{
    ucc_tl_cuda_task_t *tl_task = ucc_derived_of(task, ucc_tl_cuda_task_t);
    tl_trace(UCC_TASK_LIB(tl_task), "task: %p allgatherv_nvls_finalize", task);

    ucc_ec_destroy_event(
        tl_task->allgatherv_nvls.evt_completion, UCC_EE_CUDA_STREAM);

    ucc_tl_cuda_task_put(tl_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allgatherv_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team    = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          trank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize   = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt      = coll_args->args.dst.info_v.datatype;
    size_t              dt_size = ucc_dt_size(dt);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;
    size_t              total_count_bytes;
    size_t              offset_elements;
    size_t              count_elements;
    size_t              offset_bytes;
    size_t              count_bytes;
    ucc_rank_t          i;

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_ec_create_event(
        &task->allgatherv_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create CUDA event");
        ucc_tl_cuda_task_put(task);
        return status;
    }

    task->allgatherv_nvls.dt = dt;

    /* Get offset and count in datatype elements, then convert to bytes and
     * then uint32_t units. Datatype agnostic - we just copy raw bytes.
     * NVLS requires 16-byte alignment (4 uint32_t = 16 bytes). */
    offset_elements          = ucc_tl_cuda_allgatherv_get_offset(task, trank);
    count_elements           = ucc_tl_cuda_allgatherv_get_count(task, trank);
    offset_bytes             = offset_elements * dt_size;
    count_bytes              = count_elements * dt_size;

    /* Calculate total count in bytes */
    total_count_bytes        = 0;
    for (i = 0; i < tsize; i++) {
        total_count_bytes += ucc_tl_cuda_allgatherv_get_count(task, i) *
                             dt_size;
    }

    /* Validate total size fits within NVLS symmetric buffer */
    if (ucc_unlikely(
            total_count_bytes >
            UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_symmetric_size)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS allgatherv total size %zu bytes exceeds symmetric buffer "
            "size %zu bytes",
            total_count_bytes,
            UCC_TL_CUDA_TEAM_LIB(team)->cfg.nvls_symmetric_size);
        goto err_cleanup;
    }

    /* Convert bytes to uint32_t units for the kernel */
    task->allgatherv_nvls.offset      = offset_bytes / sizeof(uint32_t);
    task->allgatherv_nvls.count       = count_bytes / sizeof(uint32_t);
    task->allgatherv_nvls.total_count = total_count_bytes / sizeof(uint32_t);

    /* NVLS requires 16-byte alignment (4 uint32_t elements) */
    if (ucc_unlikely(task->allgatherv_nvls.offset % 4 != 0)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS requires 16-byte alignment for offset, got offset=%zu bytes "
            "(not aligned to 16 bytes). offset_elements=%zu dt_size=%zu",
            offset_bytes,
            offset_elements,
            dt_size);
        goto err_cleanup;
    }
    if (ucc_unlikely(task->allgatherv_nvls.count % 4 != 0)) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS requires 16-byte alignment for count, got count=%zu bytes "
            "(not aligned to 16 bytes). count_elements=%zu dt_size=%zu",
            count_bytes,
            count_elements,
            dt_size);
        goto err_cleanup;
    }

    task->allgatherv_nvls.mc_va   = (CUdeviceptr)TASK_SYMMETRIC_MC(task);
    task->allgatherv_nvls.uc_va   = (CUdeviceptr)TASK_SYMMETRIC_UC(task);
    task->allgatherv_nvls.coll_id = team->nvls.coll_ids[task->coll_id]++;

    task->super.post              = ucc_tl_cuda_allgatherv_nvls_start;
    task->super.triggered_post    = ucc_tl_cuda_allgatherv_nvls_triggered_post;
    task->super.progress          = ucc_tl_cuda_allgatherv_nvls_progress;
    task->super.finalize          = ucc_tl_cuda_allgatherv_nvls_finalize;

    *task_p                       = &task->super;
    return UCC_OK;

err_cleanup:
    ucc_ec_destroy_event(
        task->allgatherv_nvls.evt_completion, UCC_EE_CUDA_STREAM);
    ucc_tl_cuda_task_put(task);
    return UCC_ERR_NOT_SUPPORTED;
}
