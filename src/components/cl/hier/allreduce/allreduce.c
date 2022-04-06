/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../allreduce/allreduce.h"

ucc_base_coll_alg_info_t
    ucc_cl_hier_allreduce_algs[UCC_CL_HIER_ALLREDUCE_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLREDUCE_ALG_RAB] =
            {.id   = UCC_CL_HIER_ALLREDUCE_ALG_RAB,
             .name = "rab",
             .desc = "innode reduce, followed by inter node allreduce,"
                     " followed by innode broadcast"},
        [UCC_CL_HIER_ALLREDUCE_ALG_SPLIT_RAIL] =
            {.id   = UCC_CL_HIER_ALLREDUCE_ALG_SPLIT_RAIL,
             .name = "split_rail",
             .desc = "innode reduce scatter, followed by PPN concurrent inter"
                     " node allreduces, followed by innode allgather"},
        [UCC_CL_HIER_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};
