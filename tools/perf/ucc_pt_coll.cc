#include "ucc_pt_coll.h"

bool ucc_pt_coll::has_reduction()
{
    return has_reduction_;
}

bool ucc_pt_coll::has_inplace()
{
    return has_inplace_;
}

bool ucc_pt_coll::has_range()
{
    return has_range_;
}

bool ucc_pt_coll::has_bw()
{
    return has_bw_;
}

bool ucc_pt_coll::is_root()
{
    return is_root_;
}
