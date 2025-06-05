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

ucc_pt_coll_gatherv::ucc_pt_coll_gatherv(ucc_datatype_t dt,
                         ucc_memory_type mt, bool is_inplace,
                         bool is_persistent, int root_shift,
                         ucc_pt_comm *communicator,
                         ucc_pt_generator_base *generator)
                   : ucc_pt_coll(communicator, generator)
{
    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = false;
    root_shift_    = root_shift;

    coll_args.mask                = 0;
    coll_args.flags               = 0;
    coll_args.coll_type           = UCC_COLL_TYPE_GATHERV;
    coll_args.src.info.datatype   = dt;
    coll_args.src.info.mem_type   = mt;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;

    if (is_inplace) {
        coll_args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (is_persistent) {
        coll_args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }
}

ucc_status_t ucc_pt_coll_gatherv::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args      = test_args.coll_args;
    size_t           dt_size   = ucc_dt_size(coll_args.src.info.datatype);
    ucc_status_t st;
    bool         is_root;

    coll_args.root = test_args.coll_args.root;
    args           = coll_args;
    is_root        = (comm->get_rank() == args.root);
    if (is_root || root_shift_) {
        args.dst.info_v.counts = generator->get_dst_counts();
        args.dst.info_v.displacements = generator->get_dst_displs();
        UCCCHECK_GOTO(ucc_pt_alloc(&dst_header,
                                   generator->get_dst_count() * dt_size,
                                   args.dst.info_v.mem_type),
                      exit, st);
        args.dst.info_v.buffer = dst_header->addr;
    }

    if (!is_root || !UCC_IS_INPLACE(args) || root_shift_) {
        args.src.info.count = generator->get_src_count();
        st = ucc_pt_alloc(&src_header,
                          generator->get_src_count() * dt_size,
                          args.src.info.mem_type);
        if (UCC_OK != st) {
            std::cerr << "UCC perftest error: " << ucc_status_string(st)
                      << " in " << STR(_call) << "\n";
            if (is_root || root_shift_) {
                goto free_dst;
            } else {
                goto exit;
            }
        }
        args.src.info.buffer = src_header->addr;
    }
    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
exit:
    return st;
}

void ucc_pt_coll_gatherv::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args    = test_args.coll_args;
    bool             is_root = (comm->get_rank() == args.root);

    if (!is_root || !UCC_IS_INPLACE(args) || root_shift_) {
        ucc_pt_free(src_header);
    }
    if (is_root || root_shift_) {
        ucc_pt_free(dst_header);
    }
}
