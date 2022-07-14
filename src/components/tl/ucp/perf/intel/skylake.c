/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_SELECT_STR_SKYLAKE_40                        \
    "allreduce:0-2k:@0#allreduce:2k-inf:@1"

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_single_40, INTEL, SKYLAKE, 1, 40,
                             UCC_TL_UCP_ALLREDUCE_ALG_SELECT_STR_SKYLAKE_40);
