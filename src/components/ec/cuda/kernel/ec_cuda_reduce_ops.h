/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_REDUCE_OPS_H_
#define UCC_EC_CUDA_REDUCE_OPS_H_

extern "C" {
#include "utils/ucc_math_op.h"
}

#include "ec_cuda_half_sm52.h"
#include <cuda_bf16.h>
#include <cuComplex.h>

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

#define CUDA_REDUCE_WITH_OP_DEFAULT(NAME, _OP)                                      \
    template <typename _Type, typename _AlphaType, bool triggered>                  \
    __device__ ucc_status_t ucc_reduce_cuda_default_##NAME(                         \
        ucc_eee_task_reduce_t task, uint16_t flags)                                 \
    {                                                                               \
        size_t        count  = task.count;                                          \
        int           n_srcs = task.n_srcs;                                         \
        const _Type **s      = (const _Type **)task.srcs;                           \
        _Type *       d      = (_Type *)task.dst;                                   \
        size_t        start =                                                       \
            triggered ? threadIdx.x : threadIdx.x + blockIdx.x * blockDim.x; \
        size_t step = triggered ? blockDim.x : blockDim.x * gridDim.x;              \
        size_t i, j;                                                                \
        switch (n_srcs) {                                                           \
        case 2:                                                                     \
            for (i = start; i < count; i += step) {                                 \
                d[i] = _OP##_2(s[0][i], s[1][i]);                                   \
            }                                                                       \
            break;                                                                  \
        case 3:                                                                     \
            for (i = start; i < count; i += step) {                                 \
                d[i] = _OP##_3(s[0][i], s[1][i], s[2][i]);                          \
            }                                                                       \
            break;                                                                  \
        case 4:                                                                     \
            for (i = start; i < count; i += step) {                                 \
                d[i] = _OP##_4(s[0][i], s[1][i], s[2][i], s[3][i]);                 \
            }                                                                       \
            break;                                                                  \
        default:                                                                    \
            for (i = start; i < count; i += step) {                                 \
                d[i] = _OP(s[0][i], s[1][i]);                                       \
                for (j = 2; j < n_srcs; j++) {                                      \
                    d[i] = _OP(d[i], s[j][i]);                                      \
                }                                                                   \
            }                                                                       \
            break;                                                                  \
        }                                                                           \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                          \
            for (i = start; i < count; i += step) {                                 \
                d[i] = d[i] * (_AlphaType)task.alpha;                               \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    template <typename _Type, typename _AlphaType, bool triggered>                  \
    __global__ void UCC_REDUCE_CUDA_DEFAULT_##NAME(ucc_eee_task_reduce_t task,      \
                                                   uint16_t              flags)     \
    {                                                                               \
        ucc_reduce_cuda_default_##NAME<_Type, _AlphaType, triggered>(task,          \
                                                                     flags);        \
    }

#define CUDA_REDUCE_WITH_OP_STRIDED(NAME, _OP)                                     \
    template <typename _Type, typename _AlphaType, bool triggered>                 \
    __device__ ucc_status_t ucc_reduce_cuda_strided_##NAME(                        \
        ucc_eee_task_reduce_strided_t task, uint16_t flags)                        \
    {                                                                              \
        uint16_t     n_src2 = task.n_src2;                                         \
        size_t       count  = task.count;                                          \
        size_t       stride = task.stride;                                         \
        size_t       ld     = stride / sizeof(_Type);                              \
        const _Type *s1     = (const _Type *)task.src1;                            \
        const _Type *s2     = (const _Type *)task.src2;                            \
        _Type *      d      = (_Type *)task.dst;                                   \
        size_t       start =                                                       \
            triggered ? threadIdx.x : threadIdx.x + blockIdx.x * blockDim.x; \
        size_t step = triggered ? blockDim.x : blockDim.x * gridDim.x;             \
        size_t i, j;                                                               \
        ucc_assert(stride % sizeof(_Type) == 0);                                   \
        switch (n_src2) {                                                          \
        case 1:                                                                    \
            for (i = start; i < count; i += step) {                                \
                d[i] = _OP##_2(s1[i], s2[i]);                                      \
            }                                                                      \
            break;                                                                 \
        case 2:                                                                    \
            for (i = start; i < count; i += step) {                                \
                d[i] = _OP##_3(s1[i], s2[i], s2[i + ld]);                          \
            }                                                                      \
            break;                                                                 \
        case 3:                                                                    \
            for (i = start; i < count; i += step) {                                \
                d[i] = _OP##_4(s1[i], s2[i], s2[i + ld], s2[i + 2 * ld]);          \
            }                                                                      \
            break;                                                                 \
        default:                                                                   \
            for (i = start; i < count; i += step) {                                \
                d[i] = _OP(s1[i], s2[i]);                                          \
                for (j = 1; j < n_src2; j++) {                                     \
                    d[i] = _OP(d[i], s2[i + j * ld]);                              \
                }                                                                  \
            }                                                                      \
            break;                                                                 \
        }                                                                          \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                         \
            for (i = start; i < count; i += step) {                                \
                d[i] = d[i] * (_AlphaType)task.alpha;                              \
            }                                                                      \
        }                                                                          \
    }                                                                              \
    template <typename _Type, typename _AlphaType, bool triggered>                 \
    __global__ void UCC_REDUCE_CUDA_STRIDED_##NAME(                                \
        ucc_eee_task_reduce_strided_t task, uint16_t flags)                        \
    {                                                                              \
        ucc_reduce_cuda_strided_##NAME<_Type, _AlphaType, triggered>(task,         \
                                                                     flags);       \
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

CUDA_REDUCE_WITH_OP_DEFAULT(SUM,  DO_OP_SUM);
CUDA_REDUCE_WITH_OP_DEFAULT(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP_DEFAULT(MIN,  DO_OP_MIN);
CUDA_REDUCE_WITH_OP_DEFAULT(MAX,  DO_OP_MAX);
CUDA_REDUCE_WITH_OP_DEFAULT(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP_DEFAULT(LOR,  DO_OP_LOR);
CUDA_REDUCE_WITH_OP_DEFAULT(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP_DEFAULT(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP_DEFAULT(BOR,  DO_OP_BOR);
CUDA_REDUCE_WITH_OP_DEFAULT(BXOR, DO_OP_BXOR);

CUDA_REDUCE_WITH_OP_STRIDED(SUM,  DO_OP_SUM);
CUDA_REDUCE_WITH_OP_STRIDED(PROD, DO_OP_PROD);
CUDA_REDUCE_WITH_OP_STRIDED(MIN,  DO_OP_MIN);
CUDA_REDUCE_WITH_OP_STRIDED(MAX,  DO_OP_MAX);
CUDA_REDUCE_WITH_OP_STRIDED(LAND, DO_OP_LAND);
CUDA_REDUCE_WITH_OP_STRIDED(LOR,  DO_OP_LOR);
CUDA_REDUCE_WITH_OP_STRIDED(LXOR, DO_OP_LXOR);
CUDA_REDUCE_WITH_OP_STRIDED(BAND, DO_OP_BAND);
CUDA_REDUCE_WITH_OP_STRIDED(BOR,  DO_OP_BOR);
CUDA_REDUCE_WITH_OP_STRIDED(BXOR, DO_OP_BXOR);

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

#define DT_REDUCE_INT(_Type, _task, _op, ...)               \
    do {                                                    \
        switch (_op) {                                      \
        case UCC_OP_AVG:                                    \
        case UCC_OP_SUM:                                    \
            LAUNCH_REDUCE(SUM, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_PROD:                                   \
            LAUNCH_REDUCE(PROD, _Type, _task, __VA_ARGS__); \
            break;                                          \
        case UCC_OP_MIN:                                    \
            LAUNCH_REDUCE(MIN, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_MAX:                                    \
            LAUNCH_REDUCE(MAX, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_LAND:                                   \
            LAUNCH_REDUCE(LAND, _Type, _task, __VA_ARGS__); \
            break;                                          \
        case UCC_OP_BAND:                                   \
            LAUNCH_REDUCE(BAND, _Type, _task, __VA_ARGS__); \
            break;                                          \
        case UCC_OP_LOR:                                    \
            LAUNCH_REDUCE(LOR, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_BOR:                                    \
            LAUNCH_REDUCE(BOR, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_LXOR:                                   \
            LAUNCH_REDUCE(LXOR, _Type, _task, __VA_ARGS__); \
            break;                                          \
        case UCC_OP_BXOR:                                   \
            LAUNCH_REDUCE(BXOR, _Type, _task, __VA_ARGS__); \
            break;                                          \
        default:                                            \
            return UCC_ERR_NOT_SUPPORTED;                   \
        }                                                   \
    } while (0)

#define DT_REDUCE_FLOAT_COMPLEX(_Type, _alphaType, _task, _op, ...)       \
    do {                                                                  \
        switch (_op) {                                                    \
        case UCC_OP_AVG:                                                  \
        case UCC_OP_SUM:                                                  \
            LAUNCH_REDUCE_A(SUM, _Type, _alphaType, _task, __VA_ARGS__);  \
            break;                                                        \
        case UCC_OP_PROD:                                                 \
            LAUNCH_REDUCE_A(PROD, _Type, _alphaType, _task, __VA_ARGS__); \
            break;                                                        \
        default:                                                          \
            return UCC_ERR_NOT_SUPPORTED;                                 \
        }                                                                 \
    } while (0)

#define DT_REDUCE_FLOAT(_Type, _task, _op, ...)             \
    do {                                                    \
        switch (_op) {                                      \
        case UCC_OP_AVG:                                    \
        case UCC_OP_SUM:                                    \
            LAUNCH_REDUCE(SUM, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_PROD:                                   \
            LAUNCH_REDUCE(PROD, _Type, _task, __VA_ARGS__); \
            break;                                          \
        case UCC_OP_MIN:                                    \
            LAUNCH_REDUCE(MIN, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        case UCC_OP_MAX:                                    \
            LAUNCH_REDUCE(MAX, _Type, _task, __VA_ARGS__);  \
            break;                                          \
        default:                                            \
            return UCC_ERR_NOT_SUPPORTED;                   \
        }                                                   \
    } while (0)

#endif
