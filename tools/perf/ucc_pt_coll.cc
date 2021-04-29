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
