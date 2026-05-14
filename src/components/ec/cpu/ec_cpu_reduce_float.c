/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cpu_reduce.h"

ucc_status_t ucc_ec_cpu_reduce_float(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags)
{
    switch (task->dt) {
    case UCC_DT_FLOAT32:
#if SIZEOF_FLOAT == 4
        DO_DT_REDUCE_FLOAT(
            float, srcs, dst, task->op, task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64:
#if SIZEOF_DOUBLE == 8
        DO_DT_REDUCE_FLOAT(
            double, srcs, dst, task->op, task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT128:
#if SIZEOF_LONG_DOUBLE == 16
        DO_DT_REDUCE_FLOAT(
            long double, srcs, dst, task->op, task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_BFLOAT16:
        DO_DT_REDUCE_BFLOAT16(srcs, dst, task->op, task->count, task->n_srcs);
        break;
    default:
        ec_error(
            &ucc_ec_cpu.super,
            "unsupported reduction type (%s)",
            ucc_datatype_str(task->dt));
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}
