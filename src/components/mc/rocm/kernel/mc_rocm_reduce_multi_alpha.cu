/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_rocm.h"
#include "utils/ucc_math.h"

#include "mc_rocm_reduce_ops.h"

#define ROCM_REDUCE_ALPHA_WITH_OP(NAME, REDUCE_OP, VECTOR_OP)                  \
    template <typename T>                                                      \
    __global__ void UCC_REDUCE_ALPHA_ROCM_##NAME(                              \
        const T *s1, const T *s2, T *d, size_t size, size_t count, size_t ld,  \
        const T alpha)                                                         \
    {                                                                          \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        for (size_t i = start; i < count; i += step) {                         \
            d[i] = REDUCE_OP(s1[i], s2[i]);                                    \
            for (size_t j = 1; j < size; j++) {                                \
                d[i] = REDUCE_OP(d[i], s2[i + j * ld]);                        \
            }                                                                  \
            d[i] = VECTOR_OP(d[i], alpha);                                     \
        }                                                                      \
    }

#define ROCM_REDUCE_ALPHA_WITH_OP_SPECIALIZED(NAME, REDUCE_OP, TYPE,           \
                                              VECTOR_OP)                       \
    template <>                                                                \
    __global__ void UCC_REDUCE_ALPHA_ROCM_##NAME(                              \
        const TYPE *s1, const TYPE *s2, TYPE *d, size_t size, size_t count,    \
        size_t ld, const TYPE alpha)                                           \
    {                                                                          \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        for (size_t i = start; i < count; i += step) {                         \
            d[i] = REDUCE_OP(s1[i], s2[i]);                                    \
            for (size_t j = 1; j < size; j++) {                                \
                d[i] = REDUCE_OP(d[i], s2[i + j * ld]);                        \
            }                                                                  \
            d[i] = VECTOR_OP(d[i], alpha);                                     \
        }                                                                      \
    }

__global__ void UCC_REDUCE_ALPHA_ROCM_SUM_WITH_PROD_COMPLEX(
    const hipFloatComplex *s1, const hipFloatComplex *s2, hipFloatComplex *d,
    size_t size, size_t count, size_t ld, double alpha)
{
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step  = blockDim.x * gridDim.x;
    for (size_t i = start; i < count; i += step) {
        d[i] = DO_OP_SUM_FLOAT_COMPLEX(s1[i], s2[i]);
        for (size_t j = 1; j < size; j++) {
            d[i] = DO_OP_SUM_FLOAT_COMPLEX(d[i], s2[i + j * ld]);
        }
        d[i] = DO_OP_PROD_SCALAR_FLOAT_COMPLEX(d[i], alpha);
    }
}

__global__ void UCC_REDUCE_ALPHA_ROCM_SUM_WITH_PROD_COMPLEX(
    const hipDoubleComplex *s1, const hipDoubleComplex *s2, hipDoubleComplex *d,
    size_t size, size_t count, size_t ld, double alpha)
{
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step  = blockDim.x * gridDim.x;
    for (size_t i = start; i < count; i += step) {
        d[i] = DO_OP_SUM_DOUBLE_COMPLEX(s1[i], s2[i]);
        for (size_t j = 1; j < size; j++) {
            d[i] = DO_OP_SUM_DOUBLE_COMPLEX(d[i], s2[i + j * ld]);
        }
        d[i] = DO_OP_PROD_SCALAR_DOUBLE_COMPLEX(d[i], alpha);
    }
}

ROCM_REDUCE_ALPHA_WITH_OP(SUM_WITH_PROD, DO_OP_SUM, DO_OP_PROD)

ROCM_REDUCE_ALPHA_WITH_OP_SPECIALIZED(SUM_WITH_PROD, DO_OP_SUM_HALF, __half,
                                      DO_OP_PROD_HALF)

ROCM_REDUCE_ALPHA_WITH_OP_SPECIALIZED(SUM_WITH_PROD, DO_OP_SUM_BFLOAT16,
                                      hip_bfloat16, DO_OP_PROD_BFLOAT16)

#define LAUNCH_KERNEL(NAME, type, src1, src2, dest, size, count, ld, alpha, s, \
                      b, t)                                                    \
    do {                                                                       \
        UCC_REDUCE_ALPHA_ROCM_##NAME<type>                                     \
            <<<b, t, 0, s>>>(src1, src2, dest, size, count, ld, alpha);        \
    } while (0)

#define LAUNCH_KERNEL_COMPLEX(NAME, type, src1, src2, dest, size, count, ld,   \
                              alpha, s, b, t)                                  \
    do {                                                                       \
        UCC_REDUCE_ALPHA_ROCM_SUM_WITH_PROD_COMPLEX<<<b, t, 0, s>>>(           \
            src1, src2, dest, size, count, ld, alpha);                         \
    } while (0)


#define DT_REDUCE_FLOAT(type, reduce_op, src1_p, src2_p, dest_p, size, count,  \
                        ld, alpha, vector_op, s, b, t)                         \
    do {                                                                       \
        const type *sbuf1 = (const type *)src1_p;                              \
        const type *sbuf2 = (const type *)src2_p;                              \
        type *      dest  = (type *)dest_p;                                    \
        switch (vector_op) {                                                   \
        case UCC_OP_PROD:                                                      \
            switch (reduce_op) {                                               \
            case UCC_OP_AVG:                                                   \
                LAUNCH_KERNEL(SUM_WITH_PROD, type, sbuf1, sbuf2, dest, size,   \
                              count, ld, alpha, s, b, t);                      \
                break;                                                         \
            default:                                                           \
                mc_error(&ucc_mc_rocm.super,                                   \
                         "float dtype does not support "                       \
                         "requested reduce op: %s",                            \
                         ucc_reduction_op_str(reduce_op));                     \
                return UCC_ERR_NOT_SUPPORTED;                                  \
            }                                                                  \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_rocm.super,                                       \
                     "float dtype does not support "                           \
                     "requested vector op: %s",                                \
                     ucc_reduction_op_str(vector_op));                         \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT_COMPLEX(type, reduce_op, src1_p, src2_p, dest_p, size, \
                                count, ld, alpha, vector_op, s, b, t)          \
    do {                                                                       \
        const type *sbuf1 = (const type *)src1_p;                              \
        const type *sbuf2 = (const type *)src2_p;                              \
        type *      dest  = (type *)dest_p;                                    \
        switch (vector_op) {                                                   \
        case UCC_OP_PROD:                                                      \
            switch (reduce_op) {                                               \
            case UCC_OP_AVG:                                                   \
                UCC_REDUCE_ALPHA_ROCM_SUM_WITH_PROD_COMPLEX<<<b, t, 0, s>>>(   \
                    sbuf1, sbuf2, dest, size, count, ld, alpha);               \
                break;                                                         \
            default:                                                           \
                mc_error(&ucc_mc_rocm.super,                                   \
                         "float complex dtype does not support "               \
                         "requested reduce op: %s",                            \
                         ucc_reduction_op_str(reduce_op));                     \
                return UCC_ERR_NOT_SUPPORTED;                                  \
            }                                                                  \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_rocm.super,                                       \
                     "float complex dtype does not support "                   \
                     "requested vector op: %s",                                \
                     ucc_reduction_op_str(vector_op));                         \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t
ucc_mc_rocm_reduce_multi_alpha(const void *src1, const void *src2, void *dst,
                               size_t n_vectors, size_t count, size_t stride,
                               ucc_datatype_t dt, ucc_reduction_op_t reduce_op,
                               ucc_reduction_op_t vector_op, double alpha)
{
    size_t ld = stride / ucc_dt_size(dt);
    int    th = MC_ROCM_CONFIG->reduce_num_threads;
    unsigned long bk = (count + th - 1) / th;
    hipStream_t stream;

    UCC_MC_ROCM_INIT_STREAM();
    stream = ucc_mc_rocm.stream;
    if (MC_ROCM_CONFIG->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        bk = ucc_min(bk, MC_ROCM_CONFIG->reduce_num_blocks);
    }
    switch (dt) {
    case UCC_DT_FLOAT16:
        ucc_assert(2 == sizeof(__half));
        DT_REDUCE_FLOAT(__half, reduce_op, src1, src2, dst, n_vectors, count,
                        ld, __float2half((float)alpha), vector_op, stream, bk, th);
        break;
    case UCC_DT_FLOAT32:
        ucc_assert(4 == sizeof(float));
        DT_REDUCE_FLOAT(float, reduce_op, src1, src2, dst, n_vectors, count, ld,
                        (float)alpha, vector_op, stream, bk, th);
        break;
    case UCC_DT_FLOAT64:
        ucc_assert(8 == sizeof(double));
        DT_REDUCE_FLOAT(double, reduce_op, src1, src2, dst, n_vectors, count,
                        ld, alpha, vector_op, stream, bk, th);
        break;
    case UCC_DT_FLOAT32_COMPLEX:
        DT_REDUCE_FLOAT_COMPLEX(hipFloatComplex, reduce_op, src1, src2, dst, n_vectors,
                                count, ld, alpha, vector_op, stream, bk, th);
        break;
    case UCC_DT_FLOAT64_COMPLEX:
        DT_REDUCE_FLOAT_COMPLEX(hipDoubleComplex, reduce_op, src1, src2, dst, n_vectors,
                                count, ld, alpha, vector_op, stream, bk, th);
        break;
    case UCC_DT_BFLOAT16:
        DT_REDUCE_FLOAT(hip_bfloat16, reduce_op, src1, src2, dst, n_vectors,
                        count, ld, hip_bfloat16((float)alpha), vector_op, stream, bk, th);
        break;
    default:
        mc_error(&ucc_mc_rocm.super, "unsupported reduction type (%s)",
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
