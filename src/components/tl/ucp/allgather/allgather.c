/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgather_algs[UCC_TL_UCP_ALLGATHER_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive k-ing with arbitrary radix "},
        [UCC_TL_UCP_ALLGATHER_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_RING,
             .name = "ring",
             .desc = "O(N) Ring"},
        [UCC_TL_UCP_ALLGATHER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task)
{
    return ucc_tl_ucp_allgather_ring_init_common(task);
}
