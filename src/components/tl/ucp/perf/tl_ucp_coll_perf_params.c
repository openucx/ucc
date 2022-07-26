/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_coll_perf_params.h"

ucc_tl_ucp_perf_key_t* ucc_tl_ucp_perf_params[UCC_TL_UCP_N_PERF_PARAMS] =
{
    &intel_broadwell_single_node_28ppn,
    &intel_broadwell_2nodes_1ppn,
    &intel_broadwell_4nodes_1ppn,
    &intel_broadwell_8nodes_1ppn,
    &intel_broadwell_32nodes_1ppn,
    &intel_skylake_single_node_40ppn,
    &intel_skylake_2nodes_1ppn,
    &intel_skylake_4nodes_1ppn,
    &intel_skylake_8nodes_1ppn,
    &intel_skylake_32nodes_1ppn,
    &amd_rome_single_node_128ppn,
    &amd_rome_2nodes_1ppn,
    &amd_rome_4nodes_1ppn,
    &amd_rome_8nodes_1ppn,
    NULL
};
