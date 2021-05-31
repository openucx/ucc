#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_coll_barrier::ucc_pt_coll_barrier()
{
    has_inplace_= false;
    has_reduction_ = false;
    has_range_ = false;

    coll_args.mask = 0;
    coll_args.coll_type = UCC_COLL_TYPE_BARRIER;
}

ucc_status_t ucc_pt_coll_barrier::init_coll_args(size_t count,
                                                 ucc_coll_args_t &args)
{
    args = coll_args;
    return UCC_OK;
}

void ucc_pt_coll_barrier::free_coll_args(ucc_coll_args_t &args)
{
    return;
}

double ucc_pt_coll_barrier::get_bus_bw(double time_us)
{
    //TODO
    return 0.0;
}
