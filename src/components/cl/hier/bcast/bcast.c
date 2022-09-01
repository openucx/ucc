/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"
#include "../bcast/bcast.h"

ucc_base_coll_alg_info_t
    ucc_cl_hier_bcast_algs[UCC_CL_HIER_BCAST_ALG_LAST + 1] = {
        [UCC_CL_HIER_BCAST_ALG_2STEP] =
            {.id   = UCC_CL_HIER_BCAST_ALG_2STEP,
             .name = "2step",
             .desc = "intra-node and inter-node bcasts executed in parallel"},
        [UCC_CL_HIER_BCAST_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};
