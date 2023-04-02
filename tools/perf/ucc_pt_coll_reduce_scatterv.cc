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

ucc_pt_coll_reduce_scatterv::ucc_pt_coll_reduce_scatterv(ucc_datatype_t dt,
                        ucc_memory_type mt, ucc_reduction_op_t op,
                        bool is_inplace,
                        ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = true;
    has_reduction_ = true;
    has_range_     = true;
    has_bw_        = false;
    root_shift_    = 0;

    coll_args.coll_type = UCC_COLL_TYPE_REDUCE_SCATTERV;
    coll_args.mask = 0;
    if (is_inplace) {
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    coll_args.op                  = op;
    coll_args.src.info.datatype   = dt;
    coll_args.src.info.mem_type   = mt;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;
}

ucc_status_t ucc_pt_coll_reduce_scatterv::init_args(size_t count,
                                                    ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args    = test_args.coll_args;
    int              tsize   = comm->get_size();
    size_t           dt_size = ucc_dt_size(coll_args.dst.info_v.datatype);
    ucc_count_t     *counts;
    ucc_aint_t      *displs;
    ucc_status_t st;
    size_t       size_src, size_dst;


    args                          = coll_args;
    src_header                    = nullptr;
    dst_header                    = nullptr;
    args.dst.info_v.counts        = nullptr;
    args.dst.info_v.displacements = nullptr;

    if (UCC_IS_INPLACE(args)) {
        size_src = 0;
        size_dst = tsize * count * dt_size;
    } else {
        size_src = tsize * count * dt_size;
        size_dst = count * dt_size;
    }
    counts = (ucc_count_t*)ucc_malloc(tsize * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(counts, exit_err, st);

    displs = (ucc_aint_t*)ucc_malloc(tsize * sizeof(uint32_t), "displ buf");
    UCC_MALLOC_CHECK_GOTO(displs, free_counts, st);

    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size_dst, args.dst.info_v.mem_type),
                  free_displs, st);
    args.dst.info_v.buffer = dst_header->addr;
    if (!UCC_IS_INPLACE(args)) {
        args.src.info.count = count * tsize;
        UCCCHECK_GOTO(
            ucc_pt_alloc(&src_header, size_src, args.src.info.mem_type),
            free_dst, st);
        args.src.info.buffer = src_header->addr;
    }

    for (int i = 0; i < tsize; i++) {
        ((uint32_t*)counts)[i] = count;
        ((uint32_t*)displs)[i] = count * i;
    }

    args.dst.info_v.counts = counts;
    args.dst.info_v.displacements = displs;

    return UCC_OK;

free_dst:
    ucc_pt_free(dst_header);
free_displs:
    ucc_free(displs);
free_counts:
    ucc_free(counts);
exit_err:
    src_header                    = nullptr;
    dst_header                    = nullptr;
    args.dst.info_v.counts        = nullptr;
    args.dst.info_v.displacements = nullptr;
    return st;
}

void ucc_pt_coll_reduce_scatterv::free_args(ucc_pt_test_args_t &test_args)
{
    if (test_args.coll_args.dst.info_v.counts) {
        ucc_free(test_args.coll_args.dst.info_v.counts);
    }

    if (test_args.coll_args.dst.info_v.displacements) {
        ucc_free(test_args.coll_args.dst.info_v.displacements);
    }

    if (dst_header) {
        ucc_pt_free(dst_header);
        dst_header = nullptr;
    }
    if (src_header) {
        ucc_pt_free(src_header);
        src_header = nullptr;
    }
}
