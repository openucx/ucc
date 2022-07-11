/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALLV_H_
#define ALLTOALLV_H_

#include "../tl_cuda.h"
#include "../tl_cuda_coll.h"

ucc_status_t ucc_tl_cuda_alltoallv_ce_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_alltoallv_ce_start(ucc_coll_task_t *coll_task);

void ucc_tl_cuda_alltoallv_ce_progress(ucc_coll_task_t *coll_task);

ucc_status_t
ucc_tl_cuda_alltoallv_ce_triggered_post_setup(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_alltoallv_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *tl_team,
                                        ucc_coll_task_t     **task_p);

#endif
