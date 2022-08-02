/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "../ec_rocm.h"
#include "utils/ucc_math_op.h"
#include <inttypes.h>

#define ROCM_REDUCE_WITH_OP_DEFAULT(NAME, _OP)                                  \
    template <typename _Type, typename _AlphaType>                              \
    __global__ void UCC_REDUCE_ROCM_DEFAULT_##NAME(ucc_eee_task_reduce_t task,  \
                                                   uint16_t              flags) \
    {                                                                           \
        size_t        start  = blockIdx.x * blockDim.x + threadIdx.x;           \
        size_t        step   = blockDim.x * gridDim.x;                          \
        size_t        count  = task.count;                                      \
        int           n_srcs = task.n_srcs;                                     \
        const _Type **s      = (const _Type **)task.srcs;                       \
        _Type *       d      = (_Type *)task.dst;                               \
        size_t        i;                                                        \
                                                                                \
        switch (n_srcs) {                                                       \
        case 2:                                                                 \
            for (i = start; i < count; i += step) {                             \
                d[i] = _OP##_2(s[0][i], s[1][i]);                               \
            }                                                                   \
            break;                                                              \
        case 3:                                                                 \
            for (i = start; i < count; i += step) {                             \
                d[i] = _OP##_3(s[0][i], s[1][i], s[2][i]);                      \
            }                                                                   \
            break;                                                              \
        case 4:                                                                 \
            for (i = start; i < count; i += step) {                             \
                d[i] = _OP##_4(s[0][i], s[1][i], s[2][i], s[3][i]);             \
            }                                                                   \
            break;                                                              \
        default:                                                                \
            for (i = start; i < count; i += step) {                             \
                d[i] = _OP(s[0][i], s[1][i]);                                   \
                for (size_t j = 2; j < n_srcs; j++) {                           \
                    d[i] = _OP(d[i], s[j][i]);                                  \
                }                                                               \
            }                                                                   \
            break;                                                              \
        }                                                                       \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                      \
            for (i = start; i < count; i += step) {                             \
                d[i] = d[i] * (_AlphaType)task.alpha;                           \
            }                                                                   \
        }                                                                       \
    }

#define ROCM_REDUCE_WITH_OP_STRIDED(NAME, _OP)                                 \
    template <typename _Type, typename _AlphaType>                             \
    __global__ void UCC_REDUCE_ROCM_STRIDED_##NAME(                            \
        const _Type *s1, const _Type *s2, _Type *d, size_t count,              \
        size_t stride, uint16_t n_src2, const bool with_alpha,                 \
        const double alpha)                                                    \
    {                                                                          \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        size_t ld    = stride / sizeof(_Type);                                 \
        size_t i;                                                              \
                                                                               \
        ucc_assert(stride % sizeof(_Type) == 0);                               \
        switch (n_src2) {                                                      \
        case 1:                                                                \
            for (i = start; i < count; i += step) {                            \
                d[i] = _OP##_2(s1[i], s2[i]);                                  \
            }                                                                  \
            break;                                                             \
        case 2:                                                                \
            for (i = start; i < count; i += step) {                            \
                d[i] = _OP##_3(s1[i], s2[i], s2[i + ld]);                      \
            }                                                                  \
            break;                                                             \
        case 3:                                                                \
            for (i = start; i < count; i += step) {                            \
                d[i] = _OP##_4(s1[i], s2[i], s2[i + ld], s2[i + 2 * ld]);      \
            }                                                                  \
            break;                                                             \
        default:                                                               \
            for (i = start; i < count; i += step) {                            \
                d[i] = _OP(s1[i], s2[i]);                                      \
                for (size_t j = 1; j < n_src2; j++) {                          \
                    d[i] = _OP(d[i], s2[i + j * ld]);                          \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        if (with_alpha) {                                                      \
            for (i = start; i < count; i += step) {                            \
                d[i] = d[i] * (_AlphaType)alpha;                               \
            }                                                                  \
        }                                                                      \
    }

ROCM_REDUCE_WITH_OP_DEFAULT(SUM,  DO_OP_SUM);
ROCM_REDUCE_WITH_OP_DEFAULT(PROD, DO_OP_PROD);
ROCM_REDUCE_WITH_OP_DEFAULT(MIN,  DO_OP_MIN);
ROCM_REDUCE_WITH_OP_DEFAULT(MAX,  DO_OP_MAX);
ROCM_REDUCE_WITH_OP_DEFAULT(LAND, DO_OP_LAND);
ROCM_REDUCE_WITH_OP_DEFAULT(LOR,  DO_OP_LOR);
ROCM_REDUCE_WITH_OP_DEFAULT(LXOR, DO_OP_LXOR);
ROCM_REDUCE_WITH_OP_DEFAULT(BAND, DO_OP_BAND);
ROCM_REDUCE_WITH_OP_DEFAULT(BOR,  DO_OP_BOR);
ROCM_REDUCE_WITH_OP_DEFAULT(BXOR, DO_OP_BXOR);

ROCM_REDUCE_WITH_OP_STRIDED(SUM,  DO_OP_SUM);
ROCM_REDUCE_WITH_OP_STRIDED(PROD, DO_OP_PROD);
ROCM_REDUCE_WITH_OP_STRIDED(MIN,  DO_OP_MIN);
ROCM_REDUCE_WITH_OP_STRIDED(MAX,  DO_OP_MAX);
ROCM_REDUCE_WITH_OP_STRIDED(LAND, DO_OP_LAND);
ROCM_REDUCE_WITH_OP_STRIDED(LOR,  DO_OP_LOR);
ROCM_REDUCE_WITH_OP_STRIDED(LXOR, DO_OP_LXOR);
ROCM_REDUCE_WITH_OP_STRIDED(BAND, DO_OP_BAND);
ROCM_REDUCE_WITH_OP_STRIDED(BOR,  DO_OP_BOR);
ROCM_REDUCE_WITH_OP_STRIDED(BXOR, DO_OP_BXOR);

#define LAUNCH_KERNEL_A(NAME, type, _AlphaType, _task, s, b, t)                \
    do {                                                                       \
        if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {                 \
            UCC_REDUCE_ROCM_DEFAULT_##NAME<type, _AlphaType>                   \
                <<<b, t, 0, s>>>(_task->reduce, _task->flags);                 \
        } else {                                                               \
            ucc_eee_task_reduce_strided_t *trs = &_task->reduce_strided;       \
            UCC_REDUCE_ROCM_STRIDED_##NAME<type, _AlphaType><<<b, t, 0, s>>>(   \
                (type *)trs->src1, (type *)trs->src2, (type *)trs->dst,        \
                trs->count, trs->stride, trs->n_src2,                          \
                (bool)(_task->flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA),    \
                trs->alpha);                                                   \
        }                                                                      \
    } while (0)

#define LAUNCH_KERNEL(NAME, type,  _task, s, b, t)  \
    LAUNCH_KERNEL_A(NAME, type, type, _task, s, b, t)

#define DT_REDUCE_INT(type, _task, _op, s, b, t)                               \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_KERNEL(SUM, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL(PROD, type, _task, s, b, t);                         \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            LAUNCH_KERNEL(MIN, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            LAUNCH_KERNEL(MAX, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            LAUNCH_KERNEL(LAND, type, _task, s, b, t);                         \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            LAUNCH_KERNEL(BAND, type, _task, s, b, t);                         \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            LAUNCH_KERNEL(LOR, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            LAUNCH_KERNEL(BOR, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            LAUNCH_KERNEL(LXOR, type, _task, s, b, t);                         \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            LAUNCH_KERNEL(BXOR, type, _task, s, b, t);                         \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_rocm.super,                                       \
                     "int dtype does not support "                             \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT(type, _task, _op, s, b, t)                             \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_KERNEL(SUM, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL(PROD, type, _task, s, b, t);                         \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            LAUNCH_KERNEL(MIN, type, _task, s, b, t);                          \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            LAUNCH_KERNEL(MAX, type, _task, s, b, t);                          \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_rocm.super,                                       \
                     "float dtype does not support "                           \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT_COMPLEX(type, _alphaType, _task, _op, s, b, t)  \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_KERNEL_A(SUM, type , _alphaType, _task, s, b, t);    \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL_A(PROD, type, _alphaType, _task, s, b, t);       \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_rocm.super,                                       \
                     "float complex dtype does not support "                   \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_ec_rocm_reduce(ucc_ee_executor_task_args_t *task,
                                hipStream_t                  stream)
{
    int                th = EC_ROCM_CONFIG->reduce_num_threads;
    unsigned long      bk = EC_ROCM_CONFIG->reduce_num_blocks;
    ucc_reduction_op_t op;
    ucc_datatype_t     dt;
    size_t             count;

    if (task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        dt    = task->reduce.dt;
        count = task->reduce.count;
        op    = task->reduce.op;
    } else {
        ucc_assert(task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED);
        dt    = task->reduce_strided.dt;
        count = task->reduce_strided.count;
        op    = task->reduce_strided.op;
    }

    bk = ucc_min((count + th - 1) / th, bk);

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
    case UCC_DT_FLOAT16:
        DT_REDUCE_FLOAT(__half, task, op, stream, bk, th);
        break;
    case UCC_DT_FLOAT32:
#if SIZEOF_FLOAT == 4
        DT_REDUCE_FLOAT(float, task, op, stream, bk, th);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64:
#if SIZEOF_DOUBLE == 8
        DT_REDUCE_FLOAT(double, task, op, stream, bk, th);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT32_COMPLEX:
        DT_REDUCE_FLOAT_COMPLEX(hipFloatComplex, float, task, op, stream, bk, th);
        break;
    case UCC_DT_FLOAT64_COMPLEX:
        DT_REDUCE_FLOAT_COMPLEX(hipDoubleComplex, double, task, op, stream, bk, th);
        break;
    case UCC_DT_BFLOAT16:
        ucc_assert(2 == sizeof(hip_bfloat16));
        DT_REDUCE_FLOAT(hip_bfloat16, task, op, stream, bk, th);
        break;
    default:
        ec_error(&ucc_ec_rocm.super, "unsupported reduction type (%s)",
                 ucc_datatype_str(dt));
        return UCC_ERR_NOT_SUPPORTED;
    }
    ROCMCHECK(hipGetLastError());
    ROCMCHECK(hipStreamSynchronize(stream));
    return UCC_OK;
}
#ifdef __cplusplus
}
#endif
