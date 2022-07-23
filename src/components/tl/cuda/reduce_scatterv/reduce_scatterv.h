/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef REDUCE_SCATTERV_H_
#define REDUCE_SCATTERV_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_init(ucc_tl_cuda_task_t *task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_start(ucc_coll_task_t *task);

void ucc_tl_cuda_reduce_scatterv_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_reduce_scatterv_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *tl_team,
                                              ucc_coll_task_t **task_p);

#endif
