/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define COPY_LOOP_UNROLL                 8
#define REDUCE_LOOP_UNROLL_TRIGGERED     6
#define REDUCE_LOOP_UNROLL_INTERRUPTIBLE 1
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

#define CUDA_REDUCE_WITH_OP_CHUNK(offset, unroll, warp_size, _OP)              \
    do {                                                                       \
        const int warp =                                                       \
            triggered ? threadIdx.x / warp_size                                \
                      : (threadIdx.x + blockIdx.x * blockDim.x) / warp_size;   \
        const int    num_warps = triggered                                     \
                                     ? blockDim.x / warp_size                  \
                                     : (blockDim.x * gridDim.x) / warp_size;   \
        const int    idx       = threadIdx.x % warp_size;                      \
        const size_t num_lines =                                               \
            (count / (warp_size * unroll)) * (warp_size * unroll);             \
        _Type tmp1[unroll];                                                    \
        _Type tmp2[unroll];                                                    \
        for (line = offset + warp * warp_size * unroll + idx;                  \
             line < num_lines; line += num_warps * warp_size * unroll) {       \
            _Pragma("unroll") for (i = 0; i < unroll; i++)                     \
            {                                                                  \
                tmp1[i] = s1[line + warp_size * i];                            \
            }                                                                  \
            for (j = 0; j < MAXSRCS; j++) {                                    \
                if (j >= n_src2) {                                             \
                    break;                                                     \
                }                                                              \
                _Pragma("unroll") for (i = 0; i < unroll; i++)                 \
                {                                                              \
                    if constexpr (strided) {                                   \
                        tmp2[i] = s2[line + warp_size * i + j * ld];           \
                    } else {                                                   \
                        tmp2[i] = s[1 + j][line + warp_size * i];              \
                    }                                                          \
                }                                                              \
                _Pragma("unroll") for (i = 0; i < unroll; i++)                 \
                {                                                              \
                    tmp1[i] = _OP(tmp1[i], tmp2[i]);                           \
                }                                                              \
            }                                                                  \
            if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                 \
                _Pragma("unroll") for (i = 0; i < unroll; i++)                 \
                {                                                              \
                    tmp1[i] = tmp1[i] * (_AlphaType)task.alpha;                \
                }                                                              \
            }                                                                  \
            _Pragma("unroll") for (i = 0; i < unroll; i++)                     \
            {                                                                  \
                d[line + warp_size * i] = tmp1[i];                             \
            }                                                                  \
        }                                                                      \
    } while (0)

#define CUDA_REDUCE_WITH_OP(NAME, _OP)                                           \
    template <typename _Type, typename _AlphaType, bool triggered, int UNROLL,   \
              typename _TaskType>                                                \
    __device__ void ucc_reduce_cuda_##NAME(_TaskType task, uint16_t flags)       \
    {                                                                            \
        _Type *        d     = (_Type *)task.dst;                                \
        const size_t   count = task.count;                                       \
        constexpr bool strided =                                                 \
            std::is_same<_TaskType, ucc_eee_task_reduce_strided_t>::value;       \
        constexpr int MAXSRCS =                                                  \
            strided ? USHRT_MAX : UCC_EE_EXECUTOR_NUM_BUFS;                      \
        constexpr int       ALLOC_SIZE = strided ? 1 : UCC_EE_EXECUTOR_NUM_BUFS; \
        _Type *             s[ALLOC_SIZE];                                       \
        _Type *             s1;                                                  \
        _Type *             s2;                                                  \
        __shared__ uint16_t n_src2;                                              \
        size_t              ld;                                                  \
        size_t              i, j, line;                                          \
        if constexpr (strided) {                                                 \
            n_src2 = task.n_src2;                                                \
            s1     = (_Type *)task.src1;                                         \
            s2     = (_Type *)task.src2;                                         \
            ld     = task.stride / sizeof(_Type);                                \
            ucc_assert_system(task.stride % sizeof(_Type) == 0);                 \
        } else {                                                                 \
            memcpy(s, task.srcs, UCC_EE_EXECUTOR_NUM_BUFS * sizeof(_Type *));    \
            n_src2 = task.n_srcs - 1;                                            \
            s1     = s[0];                                                       \
        }                                                                        \
        CUDA_REDUCE_WITH_OP_CHUNK(0, UNROLL, WARP_SIZE, _OP);                    \
        CUDA_REDUCE_WITH_OP_CHUNK(                                               \
            (count / (WARP_SIZE * UNROLL)) * (WARP_SIZE * UNROLL), 1, 1, _OP);   \
    }                                                                            \
    template <typename _Type, typename _AlphaType, bool triggered, int UNROLL>   \
    __global__ void UCC_REDUCE_CUDA_DEFAULT_##NAME(ucc_eee_task_reduce_t task,   \
                                                   uint16_t              flags)  \
    {                                                                            \
        ucc_reduce_cuda_##NAME<_Type, _AlphaType, triggered, UNROLL,             \
                               ucc_eee_task_reduce_t>(task, flags);              \
    }                                                                            \
    template <typename _Type, typename _AlphaType, bool triggered, int UNROLL>   \
    __global__ void UCC_REDUCE_CUDA_STRIDED_##NAME(                              \
        ucc_eee_task_reduce_strided_t task, uint16_t flags)                      \
    {                                                                            \
        ucc_reduce_cuda_##NAME<_Type, _AlphaType, triggered, UNROLL,             \
                               ucc_eee_task_reduce_strided_t>(task, flags);      \
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
