/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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
             .desc = "intra-node reduce, followed by inter-node allreduce,"
                     " followed by intra-node broadcast"},
        [UCC_CL_HIER_ALLREDUCE_ALG_SPLIT_RAIL] =
            {.id   = UCC_CL_HIER_ALLREDUCE_ALG_SPLIT_RAIL,
             .name = "split_rail",
             .desc = "intra-node reduce_scatter, followed by PPN concurrent "
                    " inter-node allreduces, followed by intra-node allgather"},
        [UCC_CL_HIER_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};
