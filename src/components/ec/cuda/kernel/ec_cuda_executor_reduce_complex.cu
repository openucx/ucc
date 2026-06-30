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

__device__ ucc_status_t executor_reduce_complex(
    ucc_ee_executor_task_args_t *task, ucc_reduction_op_t op, ucc_datatype_t dt)
{
    switch (dt) {
    case UCC_DT_FLOAT32_COMPLEX:
#if SIZEOF_CUFLOATCOMPLEX == 8
        DT_REDUCE_FLOAT_COMPLEX(
            cuFloatComplex, float, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_FOUR);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64_COMPLEX:
#if SIZEOF_CUDOUBLECOMPLEX == 16
        DT_REDUCE_FLOAT_COMPLEX(
            cuDoubleComplex,
            double,
            task,
            op,
            REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}
