/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cpu_reduce.h"

ucc_status_t ucc_ec_cpu_reduce_int64(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags)
{
    switch (task->dt) {
    case UCC_DT_INT64:
        DO_DT_REDUCE_INT(
            int64_t, srcs, dst, task->op, task->count, task->n_srcs);
        break;
    case UCC_DT_UINT64:
        DO_DT_REDUCE_INT(
            uint64_t, srcs, dst, task->op, task->count, task->n_srcs);
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
