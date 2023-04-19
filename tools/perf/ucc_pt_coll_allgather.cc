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

ucc_pt_coll_allgather::ucc_pt_coll_allgather(ucc_datatype_t dt,
                         ucc_memory_type mt, bool is_inplace,
                         ucc_pt_comm *communicator) : ucc_pt_coll(communicator)

{
    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;
    root_shift_    = 0;

    coll_args.mask = 0;
    coll_args.coll_type = UCC_COLL_TYPE_ALLGATHER;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
    coll_args.dst.info.datatype = dt;
    coll_args.dst.info.mem_type = mt;
    if (is_inplace) {
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
}

ucc_status_t ucc_pt_coll_allgather::init_args(size_t single_rank_count,
                                              ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args     = test_args.coll_args;
    size_t           dt_size  = ucc_dt_size(coll_args.src.info.datatype);
    size_t           size_src = single_rank_count * dt_size;
    size_t           size_dst = comm->get_size() * single_rank_count * dt_size;
    ucc_status_t     st;

    args = coll_args;
    args.dst.info.count = single_rank_count * comm->get_size();
    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size_dst, args.dst.info.mem_type),
                  exit, st);
    args.dst.info.buffer = dst_header->addr;
    if (!UCC_IS_INPLACE(args)) {
        args.src.info.count = single_rank_count;
        UCCCHECK_GOTO(
            ucc_pt_alloc(&src_header, size_src, args.src.info.mem_type),
            free_dst, st);
        args.src.info.buffer = src_header->addr;
    }
    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
exit:
    return st;
}

float ucc_pt_coll_allgather::get_bw(float time_ms, int grsize,
                                    ucc_pt_test_args_t test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;
    float           N     = grsize;
    float           S     = args.dst.info.count *
                            ucc_dt_size(args.dst.info.datatype);

    return (S / time_ms) * ((N - 1) / N) / 1000.0;
}

void ucc_pt_coll_allgather::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;

    if (!UCC_IS_INPLACE(args)) {
        ucc_pt_free(src_header);
    }
    ucc_pt_free(dst_header);
}
