/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MATH_H_
#define UCC_MATH_H_

#include "config.h"
#include <ucs/sys/math.h>

#define ucc_min(_a, _b) ucs_min((_a), (_b))
#define ucc_max(_a, _b) ucs_max((_a), (_b))

static inline size_t ucc_dt_size(ucc_datatype_t dt) {
    switch(dt) {
    case UCC_DT_INT8:
    case UCC_DT_UINT8:
        return 1;
    case UCC_DT_INT16:
    case UCC_DT_UINT16:
    case UCC_DT_FLOAT16:
        return 2;
    case UCC_DT_INT32:
    case UCC_DT_UINT32:
    case UCC_DT_FLOAT32:
        return 4;
    case UCC_DT_INT64:
    case UCC_DT_UINT64:
    case UCC_DT_FLOAT64:
        return 8;
    case UCC_DT_INT128:
    case UCC_DT_UINT128:
        return 16;
    default:
        return 0;
    }
    return 0;
}

#endif
