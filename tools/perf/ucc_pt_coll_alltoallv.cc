#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>
extern "C" {
#include <core/ucc_mc.h>
}

ucc_pt_coll_alltoallv::ucc_pt_coll_alltoallv(int size, ucc_datatype_t dt,
                                             ucc_memory_type mt, bool is_inplace):
    comm_size(size)
{
    has_inplace_= true;
    has_reduction_= false;
    has_range_ = true;

    coll_args.mask = 0;
    coll_args.coll_type = UCC_COLL_TYPE_ALLTOALLV;
    coll_args.src.info_v.datatype = dt;
    coll_args.src.info_v.mem_type = mt;
    coll_args.dst.info_v.datatype = dt;
    coll_args.dst.info_v.mem_type = mt;
    if (is_inplace) {
        coll_args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        coll_args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
}

ucc_status_t ucc_pt_coll_alltoallv::init_coll_args(size_t count,
                                                   ucc_coll_args_t &args)
{
    size_t       dt_size = ucc_dt_size(coll_args.src.info_v.datatype);
    size_t       size    = comm_size * count * dt_size;
    ucc_status_t st      = UCC_OK;

    args = coll_args;
    UCCCHECK_GOTO(ucc_mc_alloc((void**)&args.src.info_v.counts,
                               comm_size * sizeof(uint32_t),
                               UCC_MEMORY_TYPE_HOST), exit, st);
    UCCCHECK_GOTO(ucc_mc_alloc((void**)&args.src.info_v.displacements,
                               comm_size * sizeof(uint32_t),
                               UCC_MEMORY_TYPE_HOST), free_src_count, st);
    UCCCHECK_GOTO(ucc_mc_alloc((void**)&args.dst.info_v.counts,
                               comm_size * sizeof(uint32_t),
                               UCC_MEMORY_TYPE_HOST), free_src_displ, st);
    UCCCHECK_GOTO(ucc_mc_alloc((void**)&args.dst.info_v.displacements,
                               comm_size * sizeof(uint32_t),
                               UCC_MEMORY_TYPE_HOST), free_dst_count, st);
    UCCCHECK_GOTO(ucc_mc_alloc(&args.dst.info_v.buffer, size,
                               args.dst.info_v.mem_type), free_dst_displ, st);
    if (!UCC_IS_INPLACE(args)) {
        UCCCHECK_GOTO(ucc_mc_alloc(&args.src.info_v.buffer, size,
                                   args.src.info_v.mem_type), free_dst, st);
    }
    for (int i = 0; i < comm_size; i++) {
        ((uint32_t*)args.src.info_v.counts)[i] = count;
        ((uint32_t*)args.src.info_v.displacements)[i] = count * i;
        ((uint32_t*)args.dst.info_v.counts)[i] = count;
        ((uint32_t*)args.dst.info_v.displacements)[i] = count * i;
    }
    return UCC_OK;
free_dst:
    ucc_mc_free(args.dst.info_v.buffer, args.dst.info_v.mem_type);
free_dst_displ:
    ucc_mc_free(args.dst.info_v.displacements, UCC_MEMORY_TYPE_HOST);
free_dst_count:
    ucc_mc_free(args.dst.info_v.counts, UCC_MEMORY_TYPE_HOST);
free_src_displ:
    ucc_mc_free(args.src.info_v.displacements, UCC_MEMORY_TYPE_HOST);
free_src_count:
    ucc_mc_free(args.src.info_v.counts, UCC_MEMORY_TYPE_HOST);
exit:
    return st;
}

void ucc_pt_coll_alltoallv::free_coll_args(ucc_coll_args_t &args)
{
    if (!UCC_IS_INPLACE(args)) {
        ucc_mc_free(args.src.info_v.buffer, args.src.info_v.mem_type);
    }
    ucc_mc_free(args.dst.info_v.buffer, args.dst.info_v.mem_type);
    ucc_mc_free(args.dst.info_v.counts, UCC_MEMORY_TYPE_HOST);
    ucc_mc_free(args.dst.info_v.displacements, UCC_MEMORY_TYPE_HOST);
    ucc_mc_free(args.src.info_v.counts, UCC_MEMORY_TYPE_HOST);
    ucc_mc_free(args.src.info_v.displacements, UCC_MEMORY_TYPE_HOST);
}

double ucc_pt_coll_alltoallv::get_bus_bw(double time_us)
{
    //TODO
    return 0.0;
}
