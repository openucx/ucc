/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef REDUCE_SCATTERV_H_
#define REDUCE_SCATTERV_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

enum
{
    UCC_TL_CUDA_REDUCE_SCATTERV_ALG_AUTO,
    UCC_TL_CUDA_REDUCE_SCATTERV_ALG_RING,
    UCC_TL_CUDA_REDUCE_SCATTERV_ALG_LINEAR,
#ifdef HAVE_NVLS
    UCC_TL_CUDA_REDUCE_SCATTERV_ALG_NVLS,
#endif /* HAVE_NVLS */
    UCC_TL_CUDA_REDUCE_SCATTERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_cuda_reduce_scatterv_algs[UCC_TL_CUDA_REDUCE_SCATTERV_ALG_LAST + 1];

#define UCC_TL_CUDA_REDUCE_SCATTERV_DEFAULT_ALG_SELECT_STR "reduce_scatterv:cuda:@0"

ucc_status_t ucc_tl_cuda_reduce_scatterv_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *tl_team,
                                              ucc_coll_task_t **task_p);

size_t ucc_tl_cuda_reduce_scatterv_get_count(const ucc_tl_cuda_task_t *task,
                                             ucc_rank_t                block);

size_t ucc_tl_cuda_reduce_scatterv_get_offset(const ucc_tl_cuda_task_t *task,
                                              ucc_rank_t                block);
/* Reduce scatterv ring */
ucc_status_t
ucc_tl_cuda_reduce_scatterv_ring_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t *     tl_team,
                                      ucc_coll_task_t **    task_p);

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_start(ucc_coll_task_t *task);

void ucc_tl_cuda_reduce_scatterv_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_finalize(ucc_coll_task_t *task);

/* Reduce scatterv linear */
ucc_status_t
ucc_tl_cuda_reduce_scatterv_linear_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *     tl_team,
                                        ucc_coll_task_t **    task_p);

ucc_status_t ucc_tl_cuda_reduce_scatterv_linear_start(ucc_coll_task_t *task);

void ucc_tl_cuda_reduce_scatterv_linear_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_linear_finalize(ucc_coll_task_t *task);

#ifdef HAVE_NVLS
/* Reduce scatterv NVLS */
ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_triggered_post(
    ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_init_common(
    ucc_tl_cuda_task_t *task, ucc_datatype_t dt, size_t offset_elements,
    size_t count_elements);

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p);

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_start(ucc_coll_task_t *task);

void ucc_tl_cuda_reduce_scatterv_nvls_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_nvls_finalize(ucc_coll_task_t *task);
#endif /* HAVE_NVLS */

static inline int ucc_tl_cuda_reduce_scatterv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_CUDA_REDUCE_SCATTERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_cuda_reduce_scatterv_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
