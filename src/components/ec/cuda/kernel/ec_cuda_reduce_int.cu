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
ucc_status_t ucc_ec_cuda_reduce_int(
    ucc_ee_executor_task_args_t *task, ucc_datatype_t dt, ucc_reduction_op_t op,
    unsigned long bk, int th, cudaStream_t stream)
{
    switch (dt) {
    case UCC_DT_INT8:
        DT_REDUCE_INT(int8_t, task, op, stream, bk, th);
        break;
    case UCC_DT_INT16:
        DT_REDUCE_INT(int16_t, task, op, stream, bk, th);
        break;
    case UCC_DT_INT32:
        DT_REDUCE_INT(int32_t, task, op, stream, bk, th);
        break;
    case UCC_DT_INT64:
        DT_REDUCE_INT(int64_t, task, op, stream, bk, th);
        break;
    case UCC_DT_UINT8:
        DT_REDUCE_INT(uint8_t, task, op, stream, bk, th);
        break;
    case UCC_DT_UINT16:
        DT_REDUCE_INT(uint16_t, task, op, stream, bk, th);
        break;
    case UCC_DT_UINT32:
        DT_REDUCE_INT(uint32_t, task, op, stream, bk, th);
        break;
    case UCC_DT_UINT64:
        DT_REDUCE_INT(uint64_t, task, op, stream, bk, th);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}
}
