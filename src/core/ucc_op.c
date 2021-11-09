/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucc_op.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"

ucc_status_t ucc_op_create_userdefined(const ucc_userdefined_op_ops_t *ops, void *context,
                                   ucc_reduction_op_t *op_p)
{
    ucc_op_userdefined_t *op_u;
    int ret;

    ret = ucc_posix_memalign((void **)&op_u,
                             ucc_max(sizeof(void *), UCC_BIT(UCC_REDUCTION_OP_SHIFT)),
                             sizeof(*op_u), "userdefined_op");
    if (ret != 0) {
        return UCC_ERR_NO_MEMORY;
    }

    op_u->ops     = *ops;
    op_u->context = context;
    *op_p     = ucc_op_from_userdefined(op_u);
    return UCC_OK;
}

void ucc_op_destroy(ucc_reduction_op_t op)
{
    ucc_op_userdefined_t *op_u;

    switch (op & UCC_REDUCTION_OP_CLASS_MASK) {
    case UCC_REDUCTION_OP_PREDEFINED:
        break;
    case UCC_REDUCTION_OP_USERDEFINED:
        op_u = ucc_op_to_userdefined(op);
        ucc_free(op_u);
        break;
    default:
        break;
    }
}
