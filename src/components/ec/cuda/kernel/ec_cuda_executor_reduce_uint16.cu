/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#include "ec_cuda_executor_reduce_dev.h"
#define UCC_EC_CUDA_REDUCE_OPS_DEVICE_ONLY
#include "ec_cuda_reduce_ops.h"
#undef UCC_EC_CUDA_REDUCE_OPS_DEVICE_ONLY

__device__ ucc_status_t executor_reduce_uint16(
    ucc_ee_executor_task_args_t *task, ucc_reduction_op_t op, ucc_datatype_t dt)
{
    switch (dt) {
    case UCC_DT_UINT16:
        DT_REDUCE_INT(uint16_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}
