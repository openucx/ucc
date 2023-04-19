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

ucc_pt_coll_scatterv::ucc_pt_coll_scatterv(ucc_datatype_t dt,
                         ucc_memory_type mt, bool is_inplace, int root_shift,
                         ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = true;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = false;
    root_shift_    = root_shift;

    coll_args.mask                = 0;
    coll_args.coll_type           = UCC_COLL_TYPE_SCATTERV;
    coll_args.src.info_v.datatype = dt;
    coll_args.src.info_v.mem_type = mt;
    coll_args.dst.info.datatype   = dt;
    coll_args.dst.info.mem_type   = mt;
    if (is_inplace) {
        coll_args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
}

ucc_status_t ucc_pt_coll_scatterv::init_args(size_t count,
                                             ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args      = test_args.coll_args;
    int              comm_size = comm->get_size();
    size_t           dt_size   = ucc_dt_size(coll_args.dst.info.datatype);
    size_t           size_dst  = count * dt_size;
    size_t           size_src  = comm_size * count * dt_size;
    ucc_status_t st;
    bool is_root;

    coll_args.root = test_args.coll_args.root;
    args           = coll_args;
    is_root        = (comm->get_rank() == args.root);
    if (is_root || root_shift_) {
        args.src.info_v.counts = (ucc_count_t *)
            ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
        UCC_MALLOC_CHECK_GOTO(args.src.info_v.counts, exit, st);
        args.src.info_v.displacements = (ucc_aint_t *)
            ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
        UCC_MALLOC_CHECK_GOTO(args.src.info_v.displacements, free_count, st);
        UCCCHECK_GOTO(
            ucc_pt_alloc(&src_header, size_src, args.src.info_v.mem_type),
            free_displ, st);
        args.src.info_v.buffer = src_header->addr;
        for (int i = 0; i < comm->get_size(); i++) {
            ((uint32_t*)args.src.info_v.counts)[i] = count;
            ((uint32_t*)args.src.info_v.displacements)[i] = count * i;
        }
    }
    if (!is_root || !UCC_IS_INPLACE(args) || root_shift_) {
        args.dst.info.count = count;
        st = ucc_pt_alloc(&dst_header, size_dst, args.dst.info.mem_type);
        if (UCC_OK != st) {
            std::cerr << "UCC perftest error: " << ucc_status_string(st)
                      << " in " << STR(_call) << "\n";
            if (is_root || root_shift_) {
                goto free_src;
            } else {
                goto exit;
            }
        }
        args.dst.info.buffer = dst_header->addr;
        return UCC_OK;
    }
free_src:
    ucc_pt_free(src_header);
free_displ:
    ucc_free(args.src.info_v.displacements);
free_count:
    ucc_free(args.src.info_v.counts);
exit:
    return st;
}

void ucc_pt_coll_scatterv::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args    = test_args.coll_args;
    bool             is_root = (comm->get_rank() == args.root);

    if (!is_root || !UCC_IS_INPLACE(args) || root_shift_) {
        ucc_pt_free(dst_header);
    }
    if (is_root || root_shift_) {
        ucc_pt_free(src_header);
        ucc_free(args.src.info_v.displacements);
        ucc_free(args.src.info_v.counts);
    }
}
