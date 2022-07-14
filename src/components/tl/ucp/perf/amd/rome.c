/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_SELECT_STR_ROME_128                            \
    "allreduce:0-1k:@0#allreduce:1k-inf:@1"

TL_UCP_PERF_KEY_DECLARE_BASE(amd_rome_single_128, AMD, ROME, 1, 128,
                             UCC_TL_UCP_ALLREDUCE_ALG_SELECT_STR_ROME_128);
