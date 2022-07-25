/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "tl_ucp.h"
#include "reduce_scatter.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_scatter_algs[UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST + 1] = {
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_RING] =
            {.id   = UCC_TL_UCP_REDUCE_SCATTER_ALG_RING,
             .name = "ring",
             .desc = "O(N) ring"},
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};
