/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))

__device__ inline void add_float4(float4 &d, const float4 &x, const float4 &y)
{
    d.x = x.x + y.x;
    d.y = x.y + y.y;
    d.z = x.z + y.z;
    d.w = x.w + y.w;
}

template <>
__global__ void
UCC_REDUCE_CUDA_MULTI_DST_SUM<float, false>(ucc_eee_task_reduce_multi_dst_t arg)
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

#define LAUNCH_REDUCE_A(NAME, type, _AlphaType, _task, s, b, t)                \
    do {                                                                       \
        if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {                 \
            UCC_REDUCE_CUDA_DEFAULT_##NAME<type, _AlphaType, false,            \
                                           REDUCE_LOOP_UNROLL_INTERRUPTIBLE>   \
                <<<b, t, 0, s>>>(_task->reduce, _task->flags);                 \
        } else if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED) {  \
            UCC_REDUCE_CUDA_STRIDED_##NAME<type, _AlphaType, false,            \
                                           REDUCE_LOOP_UNROLL_INTERRUPTIBLE>   \
                <<<b, t, 0, s>>>(_task->reduce_strided, _task->flags);         \
        } else {                                                               \
            UCC_REDUCE_CUDA_MULTI_DST_##NAME<type, false>                      \
                <<<b, t, 0, s>>>(_task->reduce_multi_dst);                     \
        }                                                                      \
    } while (0)

#define LAUNCH_REDUCE(NAME, type,  _task, s, b, t)  \
    LAUNCH_REDUCE_A(NAME, type, type, _task, s, b, t)

extern "C" {
ucc_status_t ucc_ec_cuda_reduce(
    ucc_ee_executor_task_args_t *task, unsigned num_threads,
    unsigned num_blocks, cudaStream_t stream)
{
    int                th = num_threads;
    unsigned long      bk = num_blocks;
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
    case UCC_DT_BFLOAT16:
        ucc_assert_system(2 == sizeof(__nv_bfloat16));
        DT_REDUCE_FLOAT(__nv_bfloat16, task, op, stream, bk, th);
        break;
    default:
        ec_error(&ucc_ec_cuda.super, "unsupported reduction type (%s)",
                 ucc_datatype_str(dt));
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}
}
