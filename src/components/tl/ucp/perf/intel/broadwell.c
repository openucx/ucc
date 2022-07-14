/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_SELECT_STR_BROADWELL_28                      \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_single_28, INTEL, BROADWELL,
                             1, 28,
                             UCC_TL_UCP_ALLREDUCE_ALG_SELECT_STR_BROADWELL_28);
