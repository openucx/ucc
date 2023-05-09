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

ucc_pt_coll_allgatherv::ucc_pt_coll_allgatherv(ucc_datatype_t dt,
                         ucc_memory_type mt, bool is_inplace,
                         ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = false;
    root_shift_    = 0;

    coll_args.mask = 0;
    coll_args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;
    if (is_inplace) {
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
}

ucc_status_t ucc_pt_coll_allgatherv::init_args(size_t count,
                                               ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args      = test_args.coll_args;
    int              comm_size = comm->get_size();
	size_t           dt_size   = ucc_dt_size(coll_args.src.info.datatype);
    size_t           size_src  = count * dt_size;
    size_t           size_dst  = comm_size * count * dt_size;
    ucc_status_t st;

    args = coll_args;
    args.dst.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.counts, exit, st);
    args.dst.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.displacements, free_count, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size_dst, args.dst.info_v.mem_type),
                  free_displ, st);
    args.dst.info_v.buffer = dst_header->addr;
    if (!UCC_IS_INPLACE(args)) {
        args.src.info.count = count;
        UCCCHECK_GOTO(
            ucc_pt_alloc(&src_header, size_src, args.src.info.mem_type),
            free_dst, st);
        args.src.info.buffer = src_header->addr;
    }
    for (int i = 0; i < comm_size; i++) {
        ((uint32_t*)args.dst.info_v.counts)[i] = count;
        ((uint32_t*)args.dst.info_v.displacements)[i] = count * i;
    }
    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
free_displ:
    ucc_free(args.dst.info_v.displacements);
free_count:
    ucc_free(args.dst.info_v.counts);
exit:
    return st;
}

void ucc_pt_coll_allgatherv::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;

    if (!UCC_IS_INPLACE(args)) {
        ucc_pt_free(src_header);
    }
    ucc_pt_free(dst_header);
    ucc_free(args.dst.info_v.counts);
    ucc_free(args.dst.info_v.displacements);
}
