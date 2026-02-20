/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_EXECUTOR_REDUCE_DEV_H_
#define UCC_EC_CUDA_EXECUTOR_REDUCE_DEV_H_

extern "C" {
#include "../ec_cuda.h"
}

#define LAUNCH_REDUCE_A(NAME, _Type, _AlphaType, _task, _unroll, ...)          \
    do {                                                                       \
        if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {                 \
            ucc_reduce_cuda_##NAME<                                            \
                _Type,                                                         \
                _AlphaType,                                                    \
                true,                                                          \
                false,                                                         \
                _unroll,                                                       \
                ucc_eee_task_reduce_t>(_task->reduce, _task->flags);           \
        } else {                                                               \
            ucc_reduce_cuda_##NAME<                                            \
                _Type,                                                         \
                _AlphaType,                                                    \
                true,                                                          \
                true,                                                          \
                _unroll,                                                       \
                ucc_eee_task_reduce_strided_t>(                                \
                _task->reduce_strided, _task->flags);                          \
        }                                                                      \
        return UCC_OK;                                                         \
    } while (0)

#define LAUNCH_REDUCE(NAME, _Type, _task, _unroll, ...)                        \
    LAUNCH_REDUCE_A(NAME, _Type, _Type, _task, _unroll)

__device__ ucc_status_t executor_reduce_int(
    ucc_ee_executor_task_args_t *task, ucc_reduction_op_t op,
    ucc_datatype_t dt);

__device__ ucc_status_t executor_reduce_fp(
    ucc_ee_executor_task_args_t *task, ucc_reduction_op_t op,
    ucc_datatype_t dt);

__device__ ucc_status_t executor_reduce_complex(
    ucc_ee_executor_task_args_t *task, ucc_reduction_op_t op,
    ucc_datatype_t dt);

#endif
