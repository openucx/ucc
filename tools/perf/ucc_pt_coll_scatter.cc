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

ucc_pt_coll_scatter::ucc_pt_coll_scatter(ucc_datatype_t dt,
                         ucc_memory_type mt, bool is_inplace, int root_shift,
                         ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;
    root_shift_    = root_shift;

    coll_args.mask              = 0;
    coll_args.coll_type         = UCC_COLL_TYPE_SCATTER;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
    coll_args.dst.info.datatype = dt;
    coll_args.dst.info.mem_type = mt;
    if (is_inplace) {
        coll_args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
}

ucc_status_t ucc_pt_coll_scatter::init_args(size_t single_rank_count,
                                            ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args     = test_args.coll_args;
    size_t           dt_size  = ucc_dt_size(coll_args.dst.info.datatype);
    size_t           size_dst = single_rank_count * dt_size;
    size_t           size_src = comm->get_size() * single_rank_count * dt_size;
    ucc_status_t     st_src = UCC_OK, st_dst = UCC_OK;
    bool is_root;

    coll_args.root      = test_args.coll_args.root;
    args                = coll_args;
    args.dst.info.count = single_rank_count;
    is_root             = (comm->get_rank() == args.root);
    if (is_root || root_shift_) {
        args.src.info.count = single_rank_count * comm->get_size();
        UCCCHECK_GOTO(
            ucc_pt_alloc(&src_header, size_src, args.src.info.mem_type),
            exit, st_src);
        args.src.info.buffer = src_header->addr;
    }
    if (!is_root || !UCC_IS_INPLACE(args) || root_shift_) {
        UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size_dst, args.dst.info.mem_type),
                      free_src, st_dst);
        args.dst.info.buffer = dst_header->addr;
    }
    return UCC_OK;
free_src:
    if ((is_root || root_shift_) && st_src == UCC_OK) {
        ucc_pt_free(src_header);
    }
    return st_dst;
exit:
    return st_src;
}

float ucc_pt_coll_scatter::get_bw(float time_ms, int grsize,
                                  ucc_pt_test_args_t test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;
    float            S    = args.dst.info.count * ucc_dt_size(args.dst.info.datatype);
    float            N    = grsize - 1;

    return (S * N) / time_ms / 1000.0;
}

void ucc_pt_coll_scatter::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args    = test_args.coll_args;
    bool             is_root = (comm->get_rank() == args.root);

    if (!is_root || !UCC_IS_INPLACE(args) || root_shift_) {
        ucc_pt_free(dst_header);
    }
    if (is_root || root_shift_) {
        ucc_pt_free(src_header);
    }
}
