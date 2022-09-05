/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce.h"
#include "../reduce/reduce.h"

ucc_base_coll_alg_info_t
    ucc_cl_hier_reduce_algs[UCC_CL_HIER_REDUCE_ALG_LAST + 1] = {
        [UCC_CL_HIER_REDUCE_ALG_2STEP] =
            {.id   = UCC_CL_HIER_REDUCE_ALG_2STEP,
             .name = "2step",
             .desc = "intra-node and inter-node reduces executed in parallel"},
        [UCC_CL_HIER_REDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};
