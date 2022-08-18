/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHERV_H_
#define ALLGATHERV_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum
{
    UCC_TL_CUDA_ALLGATHERV_ALG_AUTO,
    UCC_TL_CUDA_ALLGATHERV_ALG_RING,
    UCC_TL_CUDA_ALLGATHERV_ALG_LINEAR,
    UCC_TL_CUDA_ALLGATHERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_allgatherv_algs[UCC_TL_CUDA_ALLGATHERV_ALG_LAST + 1];

#define UCC_TL_CUDA_ALLGATHERV_DEFAULT_ALG_SELECT_STR "allgatherv:cuda:@0"

ucc_status_t ucc_tl_cuda_allgatherv_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *     tl_team,
                                         ucc_coll_task_t **    task_p);

size_t ucc_tl_cuda_allgatherv_get_count(const ucc_tl_cuda_task_t *task,
                                        ucc_rank_t                block);

size_t ucc_tl_cuda_allgatherv_get_offset(const ucc_tl_cuda_task_t *task,
                                         ucc_rank_t                block);

/* Allgatherv ring */
ucc_status_t ucc_tl_cuda_allgatherv_ring_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *     tl_team,
                                              ucc_coll_task_t **    task_p);

ucc_status_t ucc_tl_cuda_allgatherv_ring_start(ucc_coll_task_t *task);

void ucc_tl_cuda_allgatherv_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_allgatherv_ring_finalize(ucc_coll_task_t *task);

/* Allgatherv linear */
ucc_status_t ucc_tl_cuda_allgatherv_linear_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *     tl_team,
                                                ucc_coll_task_t **    task_p);

ucc_status_t ucc_tl_cuda_allgatherv_linear_start(ucc_coll_task_t *task);

void ucc_tl_cuda_allgatherv_linear_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_allgatherv_linear_finalize(ucc_coll_task_t *task);

static inline int ucc_tl_cuda_allgatherv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_ALLGATHERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_allgatherv_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
