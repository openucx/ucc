#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>
extern "C" {
#include <core/ucc_mc.h>
}

ucc_pt_coll_allgatherv::ucc_pt_coll_allgatherv(int size, ucc_datatype_t dt,
                                               ucc_memory_type mt,
                                               bool is_inplace):
    comm_size(size)
{
    has_inplace_= true;
    has_reduction_= false;
    has_range_ = true;

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

ucc_status_t ucc_pt_coll_allgatherv::init_coll_args(size_t count,
                                                  ucc_coll_args_t &args)
{
    size_t dt_size  = ucc_dt_size(coll_args.src.info.datatype);
    size_t size_src = count * dt_size;
    size_t size_dst = comm_size * count * dt_size;
    ucc_status_t st;

    args = coll_args;
    args.dst.info_v.counts = (ucc_count_t *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.counts, exit, st);
    args.dst.info_v.displacements = (ucc_aint_t *) ucc_malloc(comm_size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK_GOTO(args.dst.info_v.displacements, free_count, st);
    UCCCHECK_GOTO(ucc_mc_alloc(&args.dst.info_v.buffer, size_dst,
                               args.dst.info_v.mem_type), free_displ, st);
    if (!UCC_IS_INPLACE(args)) {
        args.src.info.count = count;
        UCCCHECK_GOTO(ucc_mc_alloc(&args.src.info.buffer, size_src,
                                   args.src.info.mem_type), free_dst, st);
    }
    for (int i = 0; i < comm_size; i++) {
        ((uint32_t*)args.dst.info_v.counts)[i] = count;
        ((uint32_t*)args.dst.info_v.displacements)[i] = count * i;
    }
    return UCC_OK;
free_dst:
    ucc_mc_free(args.dst.info_v.buffer, args.dst.info_v.mem_type);
free_displ:
    ucc_free(args.dst.info_v.displacements);
free_count:
    ucc_free(args.dst.info_v.counts);
exit:
    return st;
}

void ucc_pt_coll_allgatherv::free_coll_args(ucc_coll_args_t &args)
{
    if (!UCC_IS_INPLACE(args)) {
        ucc_mc_free(args.src.info.buffer, args.src.info.mem_type);
    }
    ucc_mc_free(args.dst.info_v.buffer, args.dst.info_v.mem_type);
    ucc_free(args.dst.info_v.counts);
    ucc_free(args.dst.info_v.displacements);
}

double ucc_pt_coll_allgatherv::get_bus_bw(double time_us)
{
    //TODO
    return 0.0;
}
