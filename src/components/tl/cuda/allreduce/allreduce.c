/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "tl_cuda.h"
#include "utils/arch/cuda_def.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_cuda_allreduce_algs[UCC_TL_CUDA_ALLREDUCE_ALG_LAST + 1] = {
#ifdef HAVE_NVLS
        [UCC_TL_CUDA_ALLREDUCE_ALG_NVLS] = {.id =
                                                UCC_TL_CUDA_ALLREDUCE_ALG_NVLS,
                                            .name = "nvls",
                                            .desc = "NVLINK SHARP allreduce"},
#endif /* HAVE_NVLS */
        [UCC_TL_CUDA_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_cuda_allreduce_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task_h)
{
    ucc_status_t        status  = UCC_ERR_NOT_IMPLEMENTED;
#ifdef HAVE_NVLS
    ucc_tl_cuda_team_t *tl_team = ucc_derived_of(team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;

    task = ucc_tl_cuda_task_get(tl_team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_coll_task_init(&task->super, coll_args, &tl_team->super.super);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_cuda_task_put(task);
        return status;
    }

    /* Use NVLS algorithm as default */
    status = ucc_tl_cuda_allreduce_nvls_init(coll_args, team, task_h);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_cuda_task_put(task);
    }
#else
    (void) coll_args;
    (void) team;
    (void) task_h;
#endif /* HAVE_NVLS */

    return status;
}
