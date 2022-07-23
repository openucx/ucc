/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_NCCL_COLL_H_
#define UCC_TL_NCCL_COLL_H_

#include "tl_nccl.h"

#define UCC_TL_NCCL_N_DEFAULT_ALG_SELECT_STR 1
extern const char
    *ucc_tl_nccl_default_alg_select_str[UCC_TL_NCCL_N_DEFAULT_ALG_SELECT_STR];

ucc_status_t ucc_tl_nccl_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t   coll_type,
                                        ucc_memory_type_t mem_type,
                                        ucc_base_coll_init_fn_t *init);

ucc_tl_nccl_task_t * ucc_tl_nccl_init_task(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t *team);

void ucc_tl_nccl_free_task(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                        ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_nccl_coll_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_nccl_collective_sync(ucc_tl_nccl_task_t *task,
                                         cudaStream_t stream);

ucc_status_t ucc_tl_nccl_allgather_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_allgatherv_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_allreduce_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_alltoall_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_alltoallv_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_bcast_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_reduce_scatter_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_reduce_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_barrier_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_gather_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_gatherv_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_scatter_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_scatterv_init(ucc_tl_nccl_task_t *task);

#endif
