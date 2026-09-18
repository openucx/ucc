/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cpu_reduce.h"

ucc_status_t ucc_ec_cpu_reduce(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags)
{
    switch (task->dt) {
    case UCC_DT_INT8:
    case UCC_DT_UINT8:
        return ucc_ec_cpu_reduce_int8(task, dst, srcs, flags);
    case UCC_DT_INT16:
    case UCC_DT_UINT16:
        return ucc_ec_cpu_reduce_int16(task, dst, srcs, flags);
    case UCC_DT_INT32:
    case UCC_DT_UINT32:
        return ucc_ec_cpu_reduce_int32(task, dst, srcs, flags);
    case UCC_DT_INT64:
    case UCC_DT_UINT64:
        return ucc_ec_cpu_reduce_int64(task, dst, srcs, flags);
    case UCC_DT_FLOAT32:
    case UCC_DT_FLOAT64:
    case UCC_DT_FLOAT128:
    case UCC_DT_BFLOAT16:
        return ucc_ec_cpu_reduce_float(task, dst, srcs, flags);
    case UCC_DT_FLOAT32_COMPLEX:
    case UCC_DT_FLOAT64_COMPLEX:
    case UCC_DT_FLOAT128_COMPLEX:
        return ucc_ec_cpu_reduce_complex(task, dst, srcs, flags);
    default:
        ec_error(
            &ucc_ec_cpu.super,
            "unsupported reduction type (%s)",
            ucc_datatype_str(task->dt));
        return UCC_ERR_NOT_SUPPORTED;
    }
}
