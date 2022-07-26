/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../tl_ucp_coll_perf_params.h"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_1_40                        \
    "allreduce:0-2k:@0#allreduce:2k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_2_1                         \
    "allreduce:0-32k:@0#allreduce:32k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_4_1                         \
    "allreduce:0-8k:@0#allreduce:8k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_8_1                         \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

#define UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_32_1                        \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_single_node_40ppn, INTEL, SKYLAKE,
                             1, 40, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_1_40);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_2nodes_1ppn, INTEL, SKYLAKE,
                             2, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_2_1);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_4nodes_1ppn, INTEL, SKYLAKE,
                             4, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_4_1);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_8nodes_1ppn, INTEL, SKYLAKE,
                             8, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_8_1);

TL_UCP_PERF_KEY_DECLARE_BASE(intel_skylake_32nodes_1ppn, INTEL, SKYLAKE,
                             32, 1, UCC_TL_UCP_ALLREDUCE_ALG_STR_SKYLAKE_32_1);
