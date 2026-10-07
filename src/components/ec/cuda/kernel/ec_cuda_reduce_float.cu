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

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))

__device__ inline void add_float4(float4 &d, const float4 &x, const float4 &y)
{
    d.x = x.x + y.x;
    d.y = x.y + y.y;
    d.z = x.z + y.z;
    d.w = x.w + y.w;
}

template <>
__global__ void UCC_REDUCE_CUDA_MULTI_DST_SUM<float, false>(
    ucc_eee_task_reduce_multi_dst_t arg)
{
    int    blocks_per_buf = gridDim.x / arg.n_bufs;
    int    buf_id         = blockIdx.x / blocks_per_buf;
    size_t step           = blockDim.x * blocks_per_buf;
    int    idx = threadIdx.x + (blockIdx.x % blocks_per_buf) * blockDim.x;
    int    align;

    align = align_pow2((intptr_t)arg.src1[buf_id], 16) |
            align_pow2((intptr_t)arg.src2[buf_id], 16) |
            align_pow2((intptr_t)arg.dst[buf_id], 16);

    if (align == 0) {
        /* aligned */
        size_t        count = arg.counts[buf_id] / 4;
        const float4 *s14   = (float4 *)arg.src1[buf_id];
        const float4 *s24   = (float4 *)arg.src2[buf_id];
        float4       *d4    = (float4 *)arg.dst[buf_id];

        for (size_t i = idx; i < count; i += step) {
            add_float4(d4[i], s14[i], s24[i]);
        }

        if (idx < arg.counts[buf_id] % 4) {
            size_t lidx                 = arg.counts[buf_id] - idx - 1;
            ((float *)
                 arg.dst[buf_id])[lidx] = ((float *)arg.src1[buf_id])[lidx] +
                                          ((float *)arg.src2[buf_id])[lidx];
        }
    } else {
        size_t       count = arg.counts[buf_id];
        const float *s1    = (float *)arg.src1[buf_id];
        const float *s2    = (float *)arg.src2[buf_id];
        float       *d     = (float *)arg.dst[buf_id];

        for (size_t i = idx; i < count; i += step) {
            d[i] = s1[i] + s2[i];
        }
    }
}

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
ucc_status_t ucc_ec_cuda_reduce_float(
    ucc_ee_executor_task_args_t *task, ucc_datatype_t dt, ucc_reduction_op_t op,
    unsigned long bk, int th, cudaStream_t stream)
{
    switch (dt) {
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
    case UCC_DT_BFLOAT16:
        ucc_assert_system(2 == sizeof(__nv_bfloat16));
        DT_REDUCE_FLOAT(__nv_bfloat16, task, op, stream, bk, th);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}
}
