#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_coll_bcast::ucc_pt_coll_bcast(ucc_datatype_t dt, ucc_memory_type mt,
                       ucc_pt_comm *communicator) : ucc_pt_coll(communicator)
{
    has_inplace_   = false;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;

    coll_args.mask = 0;
    coll_args.coll_type = UCC_COLL_TYPE_BCAST;
    coll_args.root = 0;
    coll_args.src.info.datatype = dt;
    coll_args.src.info.mem_type = mt;
}

ucc_status_t ucc_pt_coll_bcast::init_coll_args(size_t count,
                                                   ucc_coll_args_t &args)
{
    size_t dt_size = ucc_dt_size(coll_args.src.info.datatype);
    size_t size    = count * dt_size;
    ucc_status_t st;

    args = coll_args;
    args.src.info.count = count;
    UCCCHECK_GOTO(ucc_mc_alloc(&src_header, size, args.src.info.mem_type), exit,
                  st);
    args.src.info.buffer = src_header->addr;
exit:
    return st;
}

void ucc_pt_coll_bcast::free_coll_args(ucc_coll_args_t &args)
{
    ucc_mc_free(src_header);
}

float ucc_pt_coll_bcast::get_bw(float time_ms, int grsize, ucc_coll_args_t args)
{
    float S = args.src.info.count * ucc_dt_size(args.src.info.datatype);

    return S / time_ms / 1000.0;
}
