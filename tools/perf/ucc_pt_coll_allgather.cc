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
                         bool is_persistent, ucc_pt_map_type_t map_type,
                         ucc_pt_comm *communicator,
                         ucc_pt_generator_base *generator) : ucc_pt_coll(communicator, generator)

{
    size_t src_count_size = generator->get_src_count_max() * ucc_dt_size(dt);
    size_t dst_count_size = generator->get_dst_count_max() * ucc_dt_size(dt);
    ucc_status_t st;

    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;
    root_shift_    = 0;

    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, dst_count_size, mt),
                  exit, st);
    if (!is_inplace) {
        UCCCHECK_GOTO(ucc_pt_alloc(&src_header, src_count_size, mt),
                      exit, st);
    }

    coll_args.mask              = 0;
    coll_args.flags             = 0;
    coll_args.coll_type         = UCC_COLL_TYPE_ALLGATHER;
    coll_args.src.info.buffer   = src_header->addr;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
    coll_args.dst.info.buffer   = dst_header->addr;
    coll_args.dst.info.datatype = dt;
    coll_args.dst.info.mem_type = mt;

    if (is_inplace) {
        coll_args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (is_persistent) {
        coll_args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }

    if (map_type == UCC_PT_MAP_TYPE_LOCAL) {
        ucc_context_h        ctx = comm->get_context();
        ucc_mem_map_t        segments[1];
        ucc_mem_map_params_t mem_map_params;
        size_t               dst_memh_size, src_memh_size;

        mem_map_params.n_segments          = 1;
        mem_map_params.segments            = segments;

        mem_map_params.segments[0].address = dst_header->addr;
        mem_map_params.segments[0].len     = dst_count_size;
        UCCCHECK_GOTO(ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT,
                                  &mem_map_params, &dst_memh_size, &dst_memh),
                      exit, st);
        coll_args.dst_memh.local_memh = dst_memh;
        coll_args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;

        if (!is_inplace) {
            mem_map_params.segments[0].address = src_header->addr;
            mem_map_params.segments[0].len     = src_count_size;
            UCCCHECK_GOTO(ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT,
                                      &mem_map_params, &src_memh_size, &src_memh),
                          exit, st);
            coll_args.src_memh.local_memh = src_memh;
            coll_args.mask |= UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH;
        }
    } else if (map_type != UCC_PT_MAP_TYPE_NONE) {
        std::cerr << "Only local mapping is supported for perftest allgather"
                  << std::endl;
        goto exit;
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
    throw std::runtime_error("failed to initialize allgather arguments");
}

ucc_status_t ucc_pt_coll_allgather::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args     = test_args.coll_args;
    size_t           single_rank_count = generator->get_src_count();

    args = coll_args;
    args.dst.info.count = single_rank_count * comm->get_size();
    if (!UCC_IS_INPLACE(args)) {
        args.src.info.count = single_rank_count;
    }
    return UCC_OK;
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

ucc_pt_coll_allgather::~ucc_pt_coll_allgather()
{
    if (src_header) {
        ucc_pt_free(src_header);
    }
    if (dst_header) {
        ucc_pt_free(dst_header);
    }
    if (src_memh) {
        ucc_mem_unmap(&src_memh);
    }
    if (dst_memh) {
        ucc_mem_unmap(&dst_memh);
    }
}