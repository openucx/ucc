/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MATH_H_
#define UCC_MATH_H_

#include "config.h"
#include <ucs/sys/math.h>
#include "ucc_datastruct.h"
#include "ucc/api/ucc.h"
#define ucc_min(_a, _b) ucs_min((_a), (_b))
#define ucc_max(_a, _b) ucs_max((_a), (_b))
#define ucc_ilog2(_v)   ucs_ilog2((_v))

#define DO_OP_MAX(_v1, _v2) (_v1 > _v2 ? _v1 : _v2)
#define DO_OP_MIN(_v1, _v2) (_v1 < _v2 ? _v1 : _v2)
#define DO_OP_SUM(_v1, _v2) (_v1 + _v2)
#define DO_OP_PROD(_v1, _v2) (_v1 * _v2)
#define DO_OP_LAND(_v1, _v2) (_v1 && _v2)
#define DO_OP_BAND(_v1, _v2) (_v1 & _v2)
#define DO_OP_LOR(_v1, _v2) (_v1 || _v2)
#define DO_OP_BOR(_v1, _v2) (_v1 | _v2)
#define DO_OP_LXOR(_v1, _v2) ((!_v1) != (!_v2))
#define DO_OP_BXOR(_v1, _v2) (_v1 ^ _v2)

extern size_t ucc_dt_sizes[UCC_DT_USERDEFINED];
static inline size_t ucc_dt_size(ucc_datatype_t dt)
{
    if (dt < UCC_DT_USERDEFINED) {
        return ucc_dt_sizes[dt];
    }
    return 0;
}


#define PTR_OFFSET(_ptr, _offset)                                              \
    ((void *)((ptrdiff_t)(_ptr) + (size_t)(_offset)))

/* http://www.cse.yorku.ca/~oz/hash.html - Dan Bernstein string
   hash function */
static inline unsigned long ucc_str_hash_djb2(const char *str)
{
    unsigned long hash = 5381;
    int           c;

    while ('\0' != (c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

#endif
