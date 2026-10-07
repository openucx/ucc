/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

extern "C" {
#include "../ec_cuda.h"
}
#include "ec_cuda_reduce_ops.h"

#define LAUNCH_REDUCE_A(NAME, type, _AlphaType, _task, s, b, t)                \
    do {                                                                       \
        if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {                 \
            UCC_REDUCE_CUDA_DEFAULT_##NAME<                                    \
                type,                                                          \
                _AlphaType,                                                    \
                false,                                                         \
                REDUCE_LOOP_UNROLL_INTERRUPTIBLE>                              \
                <<<b, t, 0, s>>>(_task->reduce, _task->flags);                 \
        } else if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED) {  \
            UCC_REDUCE_CUDA_STRIDED_##NAME<                                    \
                type,                                                          \
                _AlphaType,                                                    \
                false,                                                         \
                REDUCE_LOOP_UNROLL_INTERRUPTIBLE>                              \
                <<<b, t, 0, s>>>(_task->reduce_strided, _task->flags);         \
        } else {                                                               \
            UCC_REDUCE_CUDA_MULTI_DST_##NAME<type, false>                      \
                <<<b, t, 0, s>>>(_task->reduce_multi_dst);                     \
        }                                                                      \
    } while (0)

#define LAUNCH_REDUCE(NAME, type, _task, s, b, t)                              \
    LAUNCH_REDUCE_A(NAME, type, type, _task, s, b, t)

extern "C" {
ucc_status_t ucc_ec_cuda_reduce_complex(
    ucc_ee_executor_task_args_t *task, ucc_datatype_t dt, ucc_reduction_op_t op,
    unsigned long bk, int th, cudaStream_t stream)
{
    switch (dt) {
    case UCC_DT_FLOAT32_COMPLEX:
#if SIZEOF_CUFLOATCOMPLEX == 8
        DT_REDUCE_FLOAT_COMPLEX(
            cuFloatComplex, float, task, op, stream, bk, th);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64_COMPLEX:
#if SIZEOF_CUDOUBLECOMPLEX == 16
        DT_REDUCE_FLOAT_COMPLEX(
            cuDoubleComplex, double, task, op, stream, bk, th);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}
}
