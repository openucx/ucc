/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_DT_GENERIC_H_
#define UCC_DT_GENERIC_H_
#include "config.h"
#include "ucc/api/ucc.h"

typedef struct ucc_op_userdefined {
    void                     *context;
    ucc_userdefined_op_ops_t     ops;
} ucc_op_userdefined_t;

#define UCC_OP_PREDEFINED_ID(_op) \
    ucc_ilog2((_op) >> UCC_REDUCTION_OP_SHIFT)

#define UCC_OP_IS_USERDEFINED(_op)                                    \
    (((_op) & UCC_REDUCTION_OP_CLASS_MASK) == UCC_REDUCTINO_OP_USERDEFINED)

#define UCC_OP_IS_PREDEFINED(_dt) \
    (((_dt) & UCC_REDUCTION_OP_CLASS_MASK) == UCC_REDUCTION_OP_PREDEFINED)

static inline
ucc_op_userdefined_t* ucc_op_to_userdefined(ucc_reduction_op_t op)
{
    return (ucc_op_userdefined_t*)(void*)(op & ~UCC_REDUCTION_OP_CLASS_MASK);
}


static inline
ucc_reduction_op_t ucc_op_from_userdefined(ucc_op_userdefined_t* op_u)
{
    return ((uintptr_t)op_u) | UCC_REDUCTION_OP_USERDEFINED;
}

#endif
