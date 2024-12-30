/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef BCAST_H_
#define BCAST_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum
{
    UCC_TL_CUDA_BCAST_ALG_LINEAR,
    UCC_TL_CUDA_BCAST_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_bcast_algs[UCC_TL_CUDA_BCAST_ALG_LAST + 1];

#define UCC_TL_CUDA_BCAST_DEFAULT_ALG_SELECT_STR "bcast:cuda:@0"

ucc_status_t ucc_tl_cuda_bcast_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t      *tl_team,
                                    ucc_coll_task_t     **task_p);

ucc_status_t ucc_tl_cuda_bcast_linear_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *tl_team,
                                           ucc_coll_task_t     **task_p);

static inline int ucc_tl_cuda_bcast_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_BCAST_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_bcast_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
