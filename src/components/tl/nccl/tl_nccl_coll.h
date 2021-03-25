/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_NCCL_COLL_H_
#define UCC_TL_NCCL_COLL_H_

#include "tl_nccl.h"

ucc_status_t ucc_tl_nccl_allgather_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_allgatherv_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_allreduce_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_alltoall_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_alltoallv_init(ucc_tl_nccl_task_t *task);

ucc_status_t ucc_tl_nccl_bcast_init(ucc_tl_nccl_task_t *task);

#endif
