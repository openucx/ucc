/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_1_128                       \
    "allreduce:0-1k:@0#allreduce:1k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_2_1                         \
    "allreduce:0-256k:@0#allreduce:256k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_4_1                         \
    "allreduce:0-16k:@0#allreduce:16k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_8_1                         \
    "allreduce:0-16k:@0#allreduce:16k-inf:@1"

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_single_node_128ppn, AMD, ROME, 1, 128,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_1_128);

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_2nodes_1ppn, AMD, ROME, 2, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_2_1);

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_4nodes_1ppn, AMD, ROME, 4, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_4_1);

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_8nodes_1ppn, AMD, ROME, 8, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_ROME_8_1);
