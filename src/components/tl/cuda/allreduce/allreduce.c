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
#else
        [UCC_TL_CUDA_ALLREDUCE_ALG_AUTO] = {.id =
                                                UCC_TL_CUDA_ALLREDUCE_ALG_AUTO,
                                            .name = "auto",
                                            .desc = "allreduce algorithm is not available without NVLS"},
#endif /* HAVE_NVLS */
        [UCC_TL_CUDA_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_cuda_allreduce_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task_h)
{
    ucc_status_t        status  = UCC_ERR_NOT_IMPLEMENTED;
#ifdef HAVE_NVLS
    /* Use NVLS algorithm as default */
    status = ucc_tl_cuda_allreduce_nvls_init(coll_args, team, task_h);
#else
    (void) coll_args;
    (void) team;
    (void) task_h;
#endif /* HAVE_NVLS */

    return status;
}
