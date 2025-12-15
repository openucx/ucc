/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv/reduce_scatterv.h"
#include "reduce_scatter/reduce_scatter.h"

ucc_status_t ucc_tl_cuda_reduce_scatter_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team  = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_datatype_t      dt    = coll_args->args.dst.info.datatype;
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;
    size_t              offset_elements;
    size_t              count_elements;

    if (coll_args->args.op != UCC_OP_SUM) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS reduce scatter is supported only with SUM operation");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (dt != UCC_DT_FLOAT32 && dt != UCC_DT_BFLOAT16) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "NVLS reduce scatter is supported only with float32 or bfloat16 "
            "datatype");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
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
     * For float32: 1 element = 1 uint32_t
     * For bfloat16: 2 elements = 1 uint32_t */
    offset_elements = ucc_tl_cuda_reduce_scatter_get_offset(task, trank);
    count_elements  = ucc_tl_cuda_reduce_scatter_get_count(task, trank);

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
