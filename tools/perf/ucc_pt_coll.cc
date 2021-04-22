#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>
extern "C" {
#include <core/ucc_mc.h>
}

ucc_pt_coll_allreduce::ucc_pt_coll_allreduce(ucc_datatype_t dt,
                                             ucc_memory_type mt,
                                             ucc_reduction_op_t op,
                                             bool is_inplace)
{
    coll_args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
    coll_args.mask = 0;
    if (is_inplace) {
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
    coll_args.mask |= UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
    coll_args.reduce.predefined_op = op;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
    coll_args.dst.info.mem_type = mt;
}

ucc_status_t ucc_pt_coll_allreduce::get_coll(size_t count,
                                             ucc_coll_args_t &args)
{
    size_t       dt_size = ucc_dt_size(coll_args.src.info.datatype);
    size_t       size    = count * dt_size;
    ucc_status_t st      = UCC_OK;

    args = coll_args;
    args.src.info.count = count;
    UCCCHECK_GOTO(ucc_mc_alloc(&args.dst.info.buffer, size,
                               args.dst.info.mem_type), exit, st);
    if (!UCC_IS_INPLACE(args)) {
        UCCCHECK_GOTO(ucc_mc_alloc(&args.src.info.buffer, size,
                                   args.src.info.mem_type), free_dst, st);
    }
    return UCC_OK;
free_dst:
    ucc_mc_free(args.dst.info.buffer, args.dst.info.mem_type);
exit:
    return st;
}

void ucc_pt_coll_allreduce::free_coll(ucc_coll_args_t &args)
{
    if (!UCC_IS_INPLACE(args)) {
        ucc_mc_free(args.src.info.buffer, args.src.info.mem_type);
    }
    ucc_mc_free(args.dst.info.buffer, args.dst.info.mem_type);
}

double ucc_pt_coll_allreduce::get_bus_bw(double time_us)
{
    //TODO
    return 0.0;
}
