/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef REDUCE_SCATTER_H_
#define REDUCE_SCATTER_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum
{
    UCC_TL_CUDA_REDUCE_SCATTER_ALG_AUTO,
    UCC_TL_CUDA_REDUCE_SCATTER_ALG_RING,
    UCC_TL_CUDA_REDUCE_SCATTER_ALG_LINEAR,
#ifdef HAVE_NVLS
    UCC_TL_CUDA_REDUCE_SCATTER_ALG_NVLS,
#endif /* HAVE_NVLS */
    UCC_TL_CUDA_REDUCE_SCATTER_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_reduce_scatter_algs[UCC_TL_CUDA_REDUCE_SCATTER_ALG_LAST + 1];

#define UCC_TL_CUDA_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR "reduce_scatter:cuda:@0"

size_t ucc_tl_cuda_reduce_scatter_get_count(const ucc_tl_cuda_task_t *task,
                                            ucc_rank_t                block);

size_t ucc_tl_cuda_reduce_scatter_get_offset(const ucc_tl_cuda_task_t *task,
                                             ucc_rank_t                block);

ucc_status_t ucc_tl_cuda_reduce_scatter_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *tl_team,
                                             ucc_coll_task_t **task_p);

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *     tl_team,
                                                  ucc_coll_task_t **    task_p);

ucc_status_t ucc_tl_cuda_reduce_scatter_linear_init(ucc_base_coll_args_t *coll_args,
                                                    ucc_base_team_t *     tl_team,
                                                    ucc_coll_task_t **    task_p);
#ifdef HAVE_NVLS
ucc_status_t ucc_tl_cuda_reduce_scatter_nvls_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *     tl_team,
                                                  ucc_coll_task_t **    task_p);
#endif /* HAVE_NVLS */

static inline int ucc_tl_cuda_reduce_scatter_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_REDUCE_SCATTER_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_reduce_scatter_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
