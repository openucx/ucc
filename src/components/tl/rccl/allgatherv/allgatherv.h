/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef RCCL_ALLGATHERV_H_
#define RCCL_ALLGATHERV_H_

#include "tl_rccl_coll.h"

enum {
    UCC_TL_RCCL_ALLGATHERV_ALG_P2P,
    UCC_TL_RCCL_ALLGATHERV_ALG_BCOPY,
    UCC_TL_RCCL_ALLGATHERV_ALG_BCAST,
    UCC_TL_RCCL_ALLGATHERV_ALG_LAST
};

#define UCC_TL_RCCL_ALLGATHERV_DEFAULT_ALG_SELECT_STR                          \
    "allgatherv:rocm:0-16k:@0#allgatherv:rocm:16k-1M:@1#allgatherv:rocm:1M-inf:@2"

extern ucc_base_coll_alg_info_t
             ucc_tl_rccl_allgatherv_algs[UCC_TL_RCCL_ALLGATHERV_ALG_LAST + 1];

ucc_status_t ucc_tl_rccl_allgatherv_p2p_start(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_rccl_allgatherv_p2p_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     team,
                                             ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_rccl_allgatherv_bcopy_start(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_rccl_allgatherv_bcopy_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_rccl_allgatherv_bcast_start(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_rccl_allgatherv_bcast_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h);

static inline int ucc_tl_rccl_allgatherv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_RCCL_ALLGATHERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_rccl_allgatherv_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
