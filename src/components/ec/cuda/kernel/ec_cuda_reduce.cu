/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../ec_cuda.h"
#include "utils/ucc_math_op.h"
#include <inttypes.h>
#ifdef __cplusplus
}
#endif

#include "ec_cuda_half_sm52.h"
#include "ec_cuda_reduce_ops.h"

#define CUDA_REDUCE_WITH_OP_DEFAULT(NAME, _OP)                                  \
    template <typename _Type, typename _AlphaType>                              \
    __global__ void UCC_REDUCE_CUDA_DEFAULT_##NAME(ucc_eee_task_reduce_t task,  \
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

#define CUDA_REDUCE_WITH_OP_STRIDED(NAME, _OP)                                 \
    template <typename _Type, typename _AlphaType>                             \
    __global__ void UCC_REDUCE_CUDA_STRIDED_##NAME(                            \
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

#define CUDA_REDUCE_WITH_OP_MULTI_DST(NAME, _OP)                               \
    template <typename _Type>                             \
    __global__ void UCC_REDUCE_CUDA_MULTI_DST_##NAME(                          \
        ucc_eee_task_reduce_multi_dst_t arg)                                   \
    {                                                                          \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        for (int j = 0; j < arg.n_bufs; j++) {                                 \
            size_t count = arg.counts[j];                                      \
            _Type *s2 = (_Type *)arg.src2[j];                                  \
            _Type *s1 = (_Type *)arg.src1[j];                                  \
            _Type *d  = (_Type *)arg.dst[j];                                   \
            for (size_t i = start; i < count; i += step) {                     \
                d[i] = _OP##_2(s1[i], s2[i]);                                  \
            }                                                                  \
        }                                                                      \
    }

CUDA_REDUCE_WITH_OP_DEFAULT(SUM, DO_OP_SUM);
CUDA_REDUCE_WITH_OP_DEFAULT(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP_DEFAULT(MIN, DO_OP_MIN);
CUDA_REDUCE_WITH_OP_DEFAULT(MAX, DO_OP_MAX);
CUDA_REDUCE_WITH_OP_DEFAULT(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP_DEFAULT(LOR, DO_OP_LOR);
CUDA_REDUCE_WITH_OP_DEFAULT(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP_DEFAULT(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP_DEFAULT(BOR, DO_OP_BOR);
CUDA_REDUCE_WITH_OP_DEFAULT(BXOR, DO_OP_BXOR);

CUDA_REDUCE_WITH_OP_STRIDED(SUM, DO_OP_SUM);
CUDA_REDUCE_WITH_OP_STRIDED(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP_STRIDED(MIN, DO_OP_MIN);
CUDA_REDUCE_WITH_OP_STRIDED(MAX, DO_OP_MAX);
CUDA_REDUCE_WITH_OP_STRIDED(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP_STRIDED(LOR, DO_OP_LOR);
CUDA_REDUCE_WITH_OP_STRIDED(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP_STRIDED(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP_STRIDED(BOR, DO_OP_BOR);
CUDA_REDUCE_WITH_OP_STRIDED(BXOR, DO_OP_BXOR);

CUDA_REDUCE_WITH_OP_MULTI_DST(SUM, DO_OP_SUM);
CUDA_REDUCE_WITH_OP_MULTI_DST(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP_MULTI_DST(MIN, DO_OP_MIN);
CUDA_REDUCE_WITH_OP_MULTI_DST(MAX, DO_OP_MAX);
CUDA_REDUCE_WITH_OP_MULTI_DST(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP_MULTI_DST(LOR, DO_OP_LOR);
CUDA_REDUCE_WITH_OP_MULTI_DST(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP_MULTI_DST(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP_MULTI_DST(BOR, DO_OP_BOR);
CUDA_REDUCE_WITH_OP_MULTI_DST(BXOR, DO_OP_BXOR);

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))

__device__ inline void add_float4(float4 &d, const float4 &x, const float4 &y)
{
    d.x = x.x + y.x;
    d.y = x.y + y.y;
    d.z = x.z + y.z;
    d.w = x.w + y.w;
}

template <>
__global__ void UCC_REDUCE_CUDA_MULTI_DST_SUM<float>(
        ucc_eee_task_reduce_multi_dst_t arg)
{
    int    blocks_per_buf = gridDim.x / arg.n_bufs;
    int    buf_id         = blockIdx.x / blocks_per_buf;
    size_t step           = blockDim.x * blocks_per_buf;
    int    idx            = threadIdx.x + (blockIdx.x % blocks_per_buf) * blockDim.x;
    int    align;

    align = align_pow2((intptr_t)arg.src1[buf_id], 16) |
            align_pow2((intptr_t)arg.src2[buf_id], 16) |
            align_pow2((intptr_t)arg.dst[buf_id], 16);

    if (align == 0) {
        /* aligned */
        size_t        count = arg.counts[buf_id] / 4;
        const float4 *s14   = (float4*)arg.src1[buf_id];
        const float4 *s24   = (float4*)arg.src2[buf_id];
        float4       *d4    = (float4*)arg.dst[buf_id];

        for (size_t i = idx; i < count; i += step) {
            add_float4(d4[i], s14[i], s24[i]);
        }

        if (idx < arg.counts[buf_id] % 4) {
            size_t lidx = arg.counts[buf_id] - idx - 1;
            ((float*)arg.dst[buf_id])[lidx] =
                ((float*)arg.src1[buf_id])[lidx] + ((float*)arg.src2[buf_id])[lidx];

        }
    } else {
        size_t       count = arg.counts[buf_id];
        const float *s1    = (float*)arg.src1[buf_id];
        const float *s2    = (float*)arg.src2[buf_id];
        float       *d     = (float*)arg.dst[buf_id];

        for (size_t i = idx; i < count; i += step) {
            d[i] = s1[i] + s2[i];
        }
    }
}

#define LAUNCH_KERNEL_A(NAME, type, _AlphaType, _task, s, b, t)                \
    do {                                                                       \
        if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {                 \
            UCC_REDUCE_CUDA_DEFAULT_##NAME<type, _AlphaType>                   \
                <<<b, t, 0, s>>>(_task->reduce, _task->flags);                 \
        } else if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED) {  \
            ucc_eee_task_reduce_strided_t *trs = &_task->reduce_strided;       \
            UCC_REDUCE_CUDA_STRIDED_##NAME<type, _AlphaType><<<b, t, 0, s>>>(  \
                (type *)trs->src1, (type *)trs->src2, (type *)trs->dst,        \
                trs->count, trs->stride, trs->n_src2,                          \
                (bool)(_task->flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA),    \
                trs->alpha);                                                   \
        } else {                                                               \
            UCC_REDUCE_CUDA_MULTI_DST_##NAME<type>                             \
                <<<b, t, 0, s>>>(_task->reduce_multi_dst);                     \
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
            ec_error(&ucc_ec_cuda.super,                                       \
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
            ec_error(&ucc_ec_cuda.super,                                       \
                     "float dtype does not support "                           \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT_COMPLEX(type, _alphaType, _task, _op, s, b, t)         \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_KERNEL_A(SUM, type, _alphaType, _task, s, b, t);            \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL_A(PROD, type, _alphaType, _task, s, b, t);           \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_cuda.super,                                       \
                     "float complex dtype does not support "                   \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_ec_cuda_reduce(ucc_ee_executor_task_args_t *task,
                                cudaStream_t                 stream)
{
    int                th = EC_CUDA_CONFIG->reduce_num_threads;
    unsigned long      bk = EC_CUDA_CONFIG->reduce_num_blocks;
    ucc_reduction_op_t op;
    ucc_datatype_t     dt;
    size_t             count;
    int                i;

    if (task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        count = task->reduce.count;
        dt    = task->reduce.dt;
        op    = task->reduce.op;
        bk    = ucc_min((count + th - 1) / th, bk);
    } else if (task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED) {
        count = task->reduce_strided.count;
        dt    = task->reduce_strided.dt;
        op    = task->reduce_strided.op;
        bk    = ucc_min((count + th - 1) / th, bk);
    } else {
        if (task->reduce_multi_dst.n_bufs == 0) {
            return UCC_OK;
        }
        count = 0;
        for (i = 0 ; i < task->reduce_multi_dst.n_bufs; i++) {
            count += task->reduce_multi_dst.counts[i];
        }
        dt    = task->reduce_multi_dst.dt;
        op    = task->reduce_multi_dst.op;
        bk    = 4 * task->reduce_multi_dst.n_bufs;
    }

    if (count == 0) {
        return UCC_OK;
    }

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
#if SIZEOF_CUFLOATCOMPLEX == 8
        DT_REDUCE_FLOAT_COMPLEX(cuFloatComplex, float, task, op, stream, bk, th);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64_COMPLEX:
#if SIZEOF_CUDOUBLECOMPLEX == 16
        DT_REDUCE_FLOAT_COMPLEX(cuDoubleComplex, double, task, op, stream, bk, th);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
#if CUDART_VERSION >= 11000
    case UCC_DT_BFLOAT16:
        ucc_assert(2 == sizeof(__nv_bfloat16));
        DT_REDUCE_FLOAT(__nv_bfloat16, task, op, stream, bk, th);
        break;
#endif
    default:
        ec_error(&ucc_ec_cuda.super, "unsupported reduction type (%s)",
                 ucc_datatype_str(dt));
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}
#ifdef __cplusplus
}
#endif
