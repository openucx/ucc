/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_coll_barrier::ucc_pt_coll_barrier(ucc_pt_comm *communicator,
                                         ucc_pt_generator_base *generator) :
                                          ucc_pt_coll(communicator, generator)
{
    has_inplace_   = false;
    has_reduction_ = false;
    has_range_     = false;
    has_bw_        = false;
    root_shift_    = 0;

    coll_args.mask = 0;
    coll_args.coll_type = UCC_COLL_TYPE_BARRIER;
}

ucc_status_t ucc_pt_coll_barrier::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;

    args = coll_args;
    return UCC_OK;
}

void ucc_pt_coll_barrier::free_args(ucc_pt_test_args_t &test_args)
{
    return;
}
