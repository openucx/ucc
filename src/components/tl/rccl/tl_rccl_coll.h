/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_RCCL_COLL_H_
#define UCC_TL_RCCL_COLL_H_

#include "tl_rccl.h"

#define UCC_TL_RCCL_N_DEFAULT_ALG_SELECT_STR 1
extern const char
    *ucc_tl_rccl_default_alg_select_str[UCC_TL_RCCL_N_DEFAULT_ALG_SELECT_STR];

ucc_status_t ucc_tl_rccl_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t   coll_type,
                                        ucc_memory_type_t mem_type,
                                        ucc_base_coll_init_fn_t *init);

ucc_tl_rccl_task_t * ucc_tl_rccl_init_task(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t *team);

void ucc_tl_rccl_free_task(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                        ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_rccl_coll_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_rccl_collective_sync(ucc_tl_rccl_task_t *task,
                                         hipStream_t stream);

ucc_status_t ucc_tl_rccl_allgather_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_allgatherv_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_allreduce_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_alltoall_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_alltoallv_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_bcast_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_reduce_scatter_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_reduce_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_barrier_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_gather_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_gatherv_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_scatter_init(ucc_tl_rccl_task_t *task);

ucc_status_t ucc_tl_rccl_scatterv_init(ucc_tl_rccl_task_t *task);

#endif
