/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_REDUCE_OPS_H_
#define UCC_EC_CUDA_REDUCE_OPS_H_

extern "C" {
#include "utils/ucc_math_op.h"
#include "../ec_cuda.h"
}

#include "ec_cuda_half_sm52.h"
#include <cuda_bf16.h>
#include <cuComplex.h>

#define COPY_LOOP_UNROLL                  8
#define REDUCE_LOOP_UNROLL_TRIGGERED_FOUR 4
#define REDUCE_LOOP_UNROLL_TRIGGERED_TWO  2
#define REDUCE_LOOP_UNROLL_TRIGGERED_ONE  1
#define REDUCE_LOOP_UNROLL_INTERRUPTIBLE  1
typedef int4 vectype;

__device__ inline
cuDoubleComplex operator+ (const cuDoubleComplex & first,
                           const cuDoubleComplex & second) {
    return cuCadd(first, second);
}

__device__ inline
cuDoubleComplex operator* (const cuDoubleComplex & first,
                           const cuDoubleComplex & second) {
    return cuCmul(first, second);
}

__device__ inline
cuDoubleComplex operator* (const cuDoubleComplex & first,
                           const double & second) {
    return make_cuDoubleComplex(cuCreal(first) * second,
                                cuCimag(first) * second);
}

__device__ inline
cuFloatComplex operator+ (const cuFloatComplex & first,
                          const cuFloatComplex & second) {
    return cuCaddf(first, second);
}

__device__ inline
cuFloatComplex operator* (const cuFloatComplex & first,
                          const cuFloatComplex & second) {
    return cuCmulf(first, second);
}

__device__ inline
cuFloatComplex operator* (const cuFloatComplex & first,
                          const float & second) {
    return make_cuFloatComplex(cuCrealf(first) * second,
                               cuCimagf(first) * second);
}

template <typename T, typename VecType> __device__ int ptrAlignVec(T *ptr)
{
    return (uint64_t)ptr % sizeof(VecType);
}

template <typename T, typename VecType>
__forceinline__ __device__ void LoadVec(T *d, T *s)
{
    *(reinterpret_cast<VecType *>(d)) = *(reinterpret_cast<VecType *>(s));
}

#define CUDA_REDUCE_WITH_OP_CHUNK(unroll, unroll_group_size, _OP, VecType)     \
    do {                                                                       \
        const int vectorize = sizeof(VecType) / sizeof(Type);                  \
        const int group =                                                      \
            triggered                                                          \
                ? threadIdx.x / unroll_group_size                              \
                : (threadIdx.x + blockIdx.x * blockDim.x) / unroll_group_size; \
        const int num_groups =                                                 \
            triggered ? blockDim.x / unroll_group_size                         \
                      : (blockDim.x * gridDim.x) / unroll_group_size;          \
        const int    idx = threadIdx.x % unroll_group_size;                    \
        const int    factor = unroll_group_size * unroll * vectorize;          \
        const size_t start =                                                   \
            offset + (group * unroll_group_size * unroll + idx) * vectorize;   \
        const size_t step = num_groups * factor;                               \
        const size_t end = offset + ((count - offset) / factor) * factor;      \
        Type tmp1[unroll][vectorize];                                          \
        Type tmp2[unroll][vectorize];                                          \
        ucc_assert_system(blockDim.x % unroll_group_size == 0);                \
        for (line = start; line < end; line += step) {                         \
            _Pragma("unroll") for (i = 0; i < unroll; i++)                     \
            {                                                                  \
                LoadVec<Type, VecType>(                                        \
                    tmp1[i], &s1[line + vectorize * unroll_group_size * i]);   \
            }                                                                  \
            for (j = 0; j < MAXSRCS && j < n_src2; j++) {                      \
                _Pragma("unroll") for (i = 0; i < unroll; i++)                 \
                {                                                              \
                    if (strided) {                                             \
                        LoadVec<Type, VecType>(                                \
                            tmp2[i],                                           \
                            &s2[line + vectorize * unroll_group_size * i +     \
                                j * ld]);                                      \
                    } else {                                                   \
                        LoadVec<Type, VecType>(                                \
                            tmp2[i],                                           \
                            &s[1 + j]                                          \
                              [line + vectorize * unroll_group_size * i]);     \
                    }                                                          \
                }                                                              \
                _Pragma("unroll") for (i = 0; i < unroll; i++)                 \
                {                                                              \
                    _Pragma("unroll") for (k = 0; k < vectorize; k++)          \
                    {                                                          \
                        tmp1[i][k] = _OP(tmp1[i][k], tmp2[i][k]);              \
                    }                                                          \
                }                                                              \
            }                                                                  \
            if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                 \
                _Pragma("unroll") for (i = 0; i < unroll; i++)                 \
                {                                                              \
                    _Pragma("unroll") for (k = 0; k < vectorize; k++)          \
                    {                                                          \
                        tmp1[i][k] = tmp1[i][k] * (AlphaType)task.alpha;       \
                    }                                                          \
                }                                                              \
            }                                                                  \
            _Pragma("unroll") for (i = 0; i < unroll; i++)                     \
            {                                                                  \
                LoadVec<Type, VecType>(                                        \
                    &d[line + vectorize * unroll_group_size * i], tmp1[i]);    \
            }                                                                  \
        }                                                                      \
        offset = max(offset, end);                                             \
        if (offset == count) {                                                 \
            return;                                                            \
        }                                                                      \
    } while (0)

#define CUDA_REDUCE_WITH_OP(NAME, _OP)                                          \
    template <typename Type, typename AlphaType, bool triggered, bool strided,  \
              int UNROLL, typename TaskType>                                    \
    __device__ void ucc_reduce_cuda_##NAME(TaskType task, uint16_t flags)       \
    {                                                                           \
        Type *       d          = (Type *)task.dst;                             \
        bool         alignedVec = 0;                                            \
        size_t       offset     = 0;                                            \
        const size_t count      = task.count;                                   \
        const int    MAXSRCS = strided ? USHRT_MAX : UCC_EE_EXECUTOR_NUM_BUFS;  \
        const int    ALLOC_SIZE = strided ? 1 : UCC_EE_EXECUTOR_NUM_BUFS;       \
        Type *       s[ALLOC_SIZE];                                             \
        Type *       s1;                                                        \
        Type *       s2;                                                        \
        __shared__ uint16_t n_src2;                                             \
        size_t              ld;                                                 \
        size_t              i, j, k, line;                                      \
        if (strided) {                                                          \
            ucc_eee_task_reduce_strided_t *task_strided_p =                     \
                (ucc_eee_task_reduce_strided_t *)&task;                         \
            n_src2 = task_strided_p->n_src2;                                    \
            s1     = (Type *)task_strided_p->src1;                              \
            s2     = (Type *)task_strided_p->src2;                              \
            ld     = task_strided_p->stride / sizeof(Type);                     \
            ucc_assert_system(task_strided_p->stride % sizeof(Type) == 0);      \
            alignedVec |= ptrAlignVec<Type, vectype>(s1);                       \
            alignedVec |= ptrAlignVec<Type, vectype>(s2);                       \
            alignedVec |= ((task_strided_p->stride % sizeof(vectype)) != 0);    \
        } else {                                                                \
            ucc_eee_task_reduce_t *task_default_p =                             \
                (ucc_eee_task_reduce_t *)&task;                                 \
            memcpy(s, task_default_p->srcs,                                     \
                   UCC_EE_EXECUTOR_NUM_BUFS * sizeof(Type *));                  \
            n_src2 = task_default_p->n_srcs - 1;                                \
            s1     = s[0];                                                      \
            for (int i = 0; i < MAXSRCS && i <= n_src2; i++)                    \
                alignedVec |= ptrAlignVec<Type, vectype>(s[i]);                 \
        }                                                                       \
        alignedVec |= ptrAlignVec<Type, vectype>(d);                            \
        ucc_assert_system(sizeof(vectype) % sizeof(Type) == 0);                 \
        /* Successive calls to CUDA_REDUCE_WITH_OP_CHUNK to reduce the buffer.*/\
        /* Each call enables or disables vectorization and/or loop unrollin   */\
        /* optimizations. Each one of the four calls except the last one may  */\
        /* only reduce the buffer partially and leave data remainder to be    */\
        /* treated by the subsequent calls.                                   */\
        if (triggered && alignedVec == 0) {                                     \
            /* if buffers align, use vectorized loads and unrolling */          \
            CUDA_REDUCE_WITH_OP_CHUNK(UNROLL, WARP_SIZE, _OP, vectype);         \
            /* call with vectorization but without unrolling */                 \
            CUDA_REDUCE_WITH_OP_CHUNK(1, 1, _OP, vectype);                      \
        }                                                                       \
        /* call with unrolling but without vectorization */                     \
        CUDA_REDUCE_WITH_OP_CHUNK(UNROLL, WARP_SIZE, _OP, Type);                \
        /* last call without unrolling nor vectorization */                     \
        CUDA_REDUCE_WITH_OP_CHUNK(1, 1, _OP, Type);                             \
    }                                                                           \
    template <typename Type, typename AlphaType, bool triggered, int UNROLL>    \
    __global__ void UCC_REDUCE_CUDA_DEFAULT_##NAME(ucc_eee_task_reduce_t task,  \
                                                   uint16_t              flags) \
    {                                                                           \
        ucc_reduce_cuda_##NAME<Type, AlphaType, triggered, false, UNROLL,       \
                               ucc_eee_task_reduce_t>(task, flags);             \
    }                                                                           \
    template <typename Type, typename AlphaType, bool triggered, int UNROLL>    \
    __global__ void UCC_REDUCE_CUDA_STRIDED_##NAME(                             \
        ucc_eee_task_reduce_strided_t task, uint16_t flags)                     \
    {                                                                           \
        ucc_reduce_cuda_##NAME<Type, AlphaType, triggered, true, UNROLL,        \
                               ucc_eee_task_reduce_strided_t>(task, flags);     \
    }

#define CUDA_REDUCE_WITH_OP_MULTI_DST(NAME, _OP)                               \
    template <typename _Type, bool triggered>                                  \
    __global__ void UCC_REDUCE_CUDA_MULTI_DST_##NAME(                          \
        ucc_eee_task_reduce_multi_dst_t arg)                                   \
    {                                                                          \
        size_t start =                                                         \
            triggered ? threadIdx.x : threadIdx.x + blockIdx.x * blockDim.x;   \
        size_t step = triggered ? blockDim.x : blockDim.x * gridDim.x;         \
        for (int j = 0; j < arg.n_bufs; j++) {                                 \
            size_t count = arg.counts[j];                                      \
            _Type *s2    = (_Type *)arg.src2[j];                               \
            _Type *s1    = (_Type *)arg.src1[j];                               \
            _Type *d     = (_Type *)arg.dst[j];                                \
            for (size_t i = start; i < count; i += step) {                     \
                d[i] = _OP##_2(s1[i], s2[i]);                                  \
            }                                                                  \
        }                                                                      \
    }

CUDA_REDUCE_WITH_OP(SUM, DO_OP_SUM);
CUDA_REDUCE_WITH_OP(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP(MIN, DO_OP_MIN);
CUDA_REDUCE_WITH_OP(MAX, DO_OP_MAX);
CUDA_REDUCE_WITH_OP(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP(LOR, DO_OP_LOR);
CUDA_REDUCE_WITH_OP(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP(BOR, DO_OP_BOR);
CUDA_REDUCE_WITH_OP(BXOR, DO_OP_BXOR);

CUDA_REDUCE_WITH_OP_MULTI_DST(SUM,  DO_OP_SUM);
CUDA_REDUCE_WITH_OP_MULTI_DST(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP_MULTI_DST(MIN,  DO_OP_MIN);
CUDA_REDUCE_WITH_OP_MULTI_DST(MAX,  DO_OP_MAX);
CUDA_REDUCE_WITH_OP_MULTI_DST(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP_MULTI_DST(LOR,  DO_OP_LOR);
CUDA_REDUCE_WITH_OP_MULTI_DST(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP_MULTI_DST(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP_MULTI_DST(BOR,  DO_OP_BOR);
CUDA_REDUCE_WITH_OP_MULTI_DST(BXOR, DO_OP_BXOR);

#define DT_REDUCE_INT(_Type, _task, _op, ...)                                  \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_REDUCE(SUM, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_REDUCE(PROD, _Type, _task, __VA_ARGS__);                    \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            LAUNCH_REDUCE(MIN, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            LAUNCH_REDUCE(MAX, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            LAUNCH_REDUCE(LAND, _Type, _task, __VA_ARGS__);                    \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            LAUNCH_REDUCE(BAND, _Type, _task, __VA_ARGS__);                    \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            LAUNCH_REDUCE(LOR, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            LAUNCH_REDUCE(BOR, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            LAUNCH_REDUCE(LXOR, _Type, _task, __VA_ARGS__);                    \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            LAUNCH_REDUCE(BXOR, _Type, _task, __VA_ARGS__);                    \
            break;                                                             \
        default:                                                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT_COMPLEX(_Type, _alphaType, _task, _op, ...)            \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_REDUCE_A(SUM, _Type, _alphaType, _task, __VA_ARGS__);       \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_REDUCE_A(PROD, _Type, _alphaType, _task, __VA_ARGS__);      \
            break;                                                             \
        default:                                                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT(_Type, _task, _op, ...)                                \
    do {                                                                       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_REDUCE(SUM, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_REDUCE(PROD, _Type, _task, __VA_ARGS__);                    \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            LAUNCH_REDUCE(MIN, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            LAUNCH_REDUCE(MAX, _Type, _task, __VA_ARGS__);                     \
            break;                                                             \
        default:                                                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#endif
