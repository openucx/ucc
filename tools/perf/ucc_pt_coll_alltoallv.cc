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

ucc_pt_coll_alltoallv::ucc_pt_coll_alltoallv(ucc_datatype_t dt,
                                             ucc_memory_type mt,
                                             bool is_inplace,
                                             bool is_persistent,
                                             ucc_pt_comm *communicator,
                                             ucc_pt_generator_base *generator)
                                             : ucc_pt_coll(communicator, generator)
{
    size_t src_count_max = generator->get_src_count_max();
    size_t dst_count_max = generator->get_dst_count_max();
    ucc_status_t st;

    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;
    root_shift_    = 0;


    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header,
                               dst_count_max * ucc_dt_size(dt),
                               mt),
                exit, st);

    if (!is_inplace) {
        UCCCHECK_GOTO(ucc_pt_alloc(&src_header,
                                src_count_max * ucc_dt_size(dt),
                                mt),
                      exit, st);
    }

    coll_args.mask                = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_args.coll_type           = UCC_COLL_TYPE_ALLTOALLV;
    coll_args.src.info_v.datatype = dt;
    coll_args.src.info_v.mem_type = mt;
    coll_args.src.info_v.buffer   = src_header->addr;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;
    coll_args.dst.info_v.buffer   = dst_header->addr;
    coll_args.flags               = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                                    UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
    if (is_inplace) {
        coll_args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (is_persistent) {
        coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }

    return;
exit:
    if (dst_header) {
        ucc_pt_free(dst_header);
        dst_header = NULL;
    }
    if (src_header) {
        ucc_pt_free(src_header);
        src_header = NULL;
    }
    throw std::runtime_error("failed to initialize alltoallv arguments");
}

ucc_status_t ucc_pt_coll_alltoallv::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args      = test_args.coll_args;

    args = coll_args;
    args.src.info_v.counts        = (ucc_count_t *) generator->get_src_counts();
    args.src.info_v.displacements = (ucc_aint_t *) generator->get_src_displs();
    args.dst.info_v.counts        = (ucc_count_t *) generator->get_dst_counts();
    args.dst.info_v.displacements = (ucc_aint_t *) generator->get_dst_displs();

    return UCC_OK;
}

float ucc_pt_coll_alltoallv::get_bw(float time_ms, int grsize,
                                    ucc_pt_test_args_t test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;
    float            N    = grsize;
    float            S    = 0;
    size_t src_size = 0, dst_size = 0;


    for (int i = 0; i < grsize; i++) {
        src_size += ucc_coll_args_get_count(&args, args.src.info_v.counts, i);
        dst_size += ucc_coll_args_get_count(&args, args.dst.info_v.counts, i);
    }
    src_size *= ucc_dt_size(args.src.info_v.datatype);
    dst_size *= ucc_dt_size(args.dst.info_v.datatype);
    S = src_size > dst_size ? src_size : dst_size;

    return (S / time_ms) * ((N - 1) / N) / 1000.0;
}

ucc_pt_coll_alltoallv::~ucc_pt_coll_alltoallv()
{
    if (src_header) {
        ucc_pt_free(src_header);
    }
    if (dst_header) {
        ucc_pt_free(dst_header);
    }
}