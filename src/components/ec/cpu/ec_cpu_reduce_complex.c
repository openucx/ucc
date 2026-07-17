/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cpu_reduce.h"

ucc_status_t ucc_ec_cpu_reduce_complex(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags)
{
    switch (task->dt) {
    case UCC_DT_FLOAT32_COMPLEX:
#if SIZEOF_FLOAT__COMPLEX == 8
        DO_DT_REDUCE_FLOAT_COMPLEX(
            float complex, srcs, dst, task->op, task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64_COMPLEX:
#if SIZEOF_DOUBLE__COMPLEX == 16
        DO_DT_REDUCE_FLOAT_COMPLEX(
            double complex, srcs, dst, task->op, task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT128_COMPLEX:
#if SIZEOF_LONG_DOUBLE__COMPLEX == 32
        DO_DT_REDUCE_FLOAT_COMPLEX(
            long double complex,
            srcs,
            dst,
            task->op,
            task->count,
            task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    default:
        ec_error(
            &ucc_ec_cpu.super,
            "unsupported reduction type (%s)",
            ucc_datatype_str(task->dt));
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}
