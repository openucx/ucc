/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum {
#ifdef HAVE_NVLS
    UCC_TL_CUDA_ALLREDUCE_ALG_NVLS,
#else
    UCC_TL_CUDA_ALLREDUCE_ALG_AUTO,
#endif /* HAVE_NVLS */
    UCC_TL_CUDA_ALLREDUCE_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_allreduce_algs[UCC_TL_CUDA_ALLREDUCE_ALG_LAST + 1];

#ifdef HAVE_NVLS
#define UCC_TL_CUDA_ALLREDUCE_DEFAULT_ALG_SELECT_STR "allreduce:cuda:@0"
#else
#define UCC_TL_CUDA_ALLREDUCE_DEFAULT_ALG_SELECT_STR ""
#endif /* HAVE_NVLS */

ucc_status_t ucc_tl_cuda_allreduce_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_cuda_allreduce_nvls_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *team,
                                             ucc_coll_task_t     **task_h);

static inline int ucc_tl_cuda_allreduce_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_ALLREDUCE_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_allreduce_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
