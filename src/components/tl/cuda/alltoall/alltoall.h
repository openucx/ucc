/**
 * Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_H_
#define ALLTOALL_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"
#include "components/base/ucc_base_iface.h"

enum {
    UCC_TL_CUDA_ALLTOALL_ALG_CE   = 0,
    UCC_TL_CUDA_ALLTOALL_ALG_PUSH = 1,
    UCC_TL_CUDA_ALLTOALL_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_alltoall_algs[UCC_TL_CUDA_ALLTOALL_ALG_LAST + 1];

static inline int ucc_tl_cuda_alltoall_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_ALLTOALL_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_alltoall_algs[i].name))
            return i;
    }
    return -1;
}

ucc_status_t ucc_tl_cuda_alltoall_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t      *tl_team,
                                       ucc_coll_task_t     **task_p);

ucc_status_t ucc_tl_cuda_alltoall_push_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t      *tl_team,
                                            ucc_coll_task_t     **task_p);

#endif
