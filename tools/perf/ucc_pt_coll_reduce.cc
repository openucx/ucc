#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_coll_reduce::ucc_pt_coll_reduce(ucc_datatype_t dt, ucc_memory_type mt,
                        ucc_reduction_op_t op, bool is_inplace,
                        ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = true;
    has_reduction_ = true;
    has_range_     = true;
    has_bw_        = true;

    coll_args.coll_type = UCC_COLL_TYPE_REDUCE;
    coll_args.mask = 0;
    if (is_inplace) {
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    coll_args.op   = op;
    coll_args.root = 0;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
    coll_args.dst.info.datatype = dt;
    coll_args.dst.info.mem_type = mt;
}

ucc_status_t ucc_pt_coll_reduce::init_args(size_t count,
                                           ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args    = test_args.coll_args;
    size_t           dt_size = ucc_dt_size(coll_args.src.info.datatype);
    size_t           size    = count * dt_size;
    ucc_status_t st_src, st_dst;

    args = coll_args;
    args.src.info.count = count;
    args.dst.info.count = count;
    bool is_root = (comm->get_rank() == args.root);
    if (is_root) {
        UCCCHECK_GOTO(ucc_mc_alloc(&dst_header, size, args.dst.info.mem_type),
                      exit, st_dst);
        args.dst.info.buffer = dst_header->addr;
    }
    if (!is_root || !UCC_IS_INPLACE(args)) {
        UCCCHECK_GOTO(ucc_mc_alloc(&src_header, size, args.src.info.mem_type),
                      free_dst, st_src);
        args.src.info.buffer = src_header->addr;
    }
    return UCC_OK;
free_dst:
    if (is_root && st_dst == UCC_OK) {
        ucc_mc_free(dst_header);
    }
    return st_src;
exit:
    return st_dst;
}

void ucc_pt_coll_reduce::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_coll_args_t &args    = test_args.coll_args;
    bool             is_root = (comm->get_rank() == args.root);

    if (!is_root || !UCC_IS_INPLACE(args)) {
        ucc_mc_free(src_header);
    }
    if (is_root) {
        ucc_mc_free(dst_header);
    }
}

float ucc_pt_coll_reduce::get_bw(float time_ms, int grsize,
                                 ucc_pt_test_args_t test_args)
{
    ucc_coll_args_t &args = test_args.coll_args;
    float            S    = args.src.info.count *
                            ucc_dt_size(args.src.info.datatype);

    return S / time_ms / 1000.0;
}
