/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_1_28                    \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_2_1                     \
    "allreduce:0-256k:@0#allreduce:256k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_4_1                     \
    "allreduce:0-2k:@0#allreduce:2k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_8_1                     \
    "allreduce:0-256:@0#allreduce:256-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_32_1                    \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1" //TODO: run perf jazz32

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_single_node_28ppn, INTEL,
                             BROADWELL, 1, 28,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_1_28);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_2nodes_1ppn, INTEL, BROADWELL,
                             2, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_2_1);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_4nodes_1ppn, INTEL, BROADWELL,
                             4, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_4_1);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_8nodes_1ppn, INTEL, BROADWELL,
                             8, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_8_1);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_broadwell_32nodes_1ppn, INTEL, BROADWELL,
                             32, 1,
                             UCC_TL_UCP_ALLREDUCE_ALG_STR_BROADWELL_32_1);
