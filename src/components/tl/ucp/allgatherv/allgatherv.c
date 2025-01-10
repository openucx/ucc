/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgatherv_algs[UCC_TL_UCP_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLGATHERV_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLGATHERV_ALG_RING,
             .name = "ring",
             .desc = "O(N) Ring"},
        [UCC_TL_UCP_ALLGATHERV_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLGATHERV_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive k-ing with arbitrary radix"},
        [UCC_TL_UCP_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task)
{
    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    return ucc_tl_ucp_allgatherv_ring_init_common(task);
}
