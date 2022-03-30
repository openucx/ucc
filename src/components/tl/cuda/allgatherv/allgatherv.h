/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHERV_H_
#define ALLGATHERV_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"


ucc_status_t ucc_tl_cuda_allgatherv_ring_init(ucc_tl_cuda_task_t *task);

ucc_status_t ucc_tl_cuda_allgatherv_ring_start(ucc_coll_task_t *task);

void ucc_tl_cuda_allgatherv_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_allgatherv_ring_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_cuda_allgatherv_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *tl_team,
                                         ucc_coll_task_t **task_p);

#endif
