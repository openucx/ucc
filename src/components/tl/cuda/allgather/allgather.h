/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHER_H_
#define ALLGATHER_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum
{
    UCC_TL_CUDA_ALLGATHER_ALG_AUTO,
    UCC_TL_CUDA_ALLGATHER_ALG_RING,
    UCC_TL_CUDA_ALLGATHER_ALG_LINEAR,
    UCC_TL_CUDA_ALLGATHER_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_allgather_algs[UCC_TL_CUDA_ALLGATHER_ALG_LAST + 1];

#define UCC_TL_CUDA_ALLGATHER_DEFAULT_ALG_SELECT_STR "allgather:cuda:@0"

size_t ucc_tl_cuda_allgather_get_count(const ucc_tl_cuda_task_t *task,
                                       ucc_rank_t                block);

size_t ucc_tl_cuda_allgather_get_offset(const ucc_tl_cuda_task_t *task,
                                        ucc_rank_t                block);

ucc_status_t ucc_tl_cuda_allgather_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *tl_team,
                                        ucc_coll_task_t **task_p);

ucc_status_t ucc_tl_cuda_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     tl_team,
                                             ucc_coll_task_t **    task_p);

ucc_status_t ucc_tl_cuda_allgather_linear_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     tl_team,
                                               ucc_coll_task_t **    task_p);

static inline int ucc_tl_cuda_allgather_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_ALLGATHER_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_allgather_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
