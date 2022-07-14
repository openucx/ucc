/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_coll_perf_params.h"

ucc_tl_ucp_perf_key_t* ucc_tl_ucp_perf_params[UCC_TL_UCP_N_PERF_PARAMS] =
{
    &intel_broadwell_single_28,
    &intel_skylake_single_40,
    &amd_rome_single_128,
    NULL
};
