/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_rocm.h"
#include "utils/ucc_math.h"

#include "mc_rocm_reduce_ops.h"

#define ROCM_REDUCE_WITH_OP(NAME, OP)                                          \
template <typename T>                                                          \
__global__ void UCC_REDUCE_ROCM_ ## NAME (const T *s1, const T *s2, T *d,      \
                                          size_t count)                        \
{                                                                              \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        for (size_t i = start; i < count; i+=step) {                           \
            d[i] = OP(s1[i], s2[i]);                                           \
        }                                                                      \
}                                                                              \

#define ROCM_REDUCE_WITH_OP_SPECIALIZED(NAME, OP, TYPE)                        \
template <>                                                                    \
__global__ void UCC_REDUCE_ROCM_ ## NAME (const TYPE *s1, const TYPE *s2,      \
                                          TYPE *d, size_t count)               \
{                                                                              \
        size_t start = blockIdx.x * blockDim.x + threadIdx.x;                  \
        size_t step  = blockDim.x * gridDim.x;                                 \
        for (size_t i = start; i < count; i+=step) {                           \
            d[i] = OP(s1[i], s2[i]);                                           \
        }                                                                      \
}                                                                              \

ROCM_REDUCE_WITH_OP(MAX,  DO_OP_MAX)
ROCM_REDUCE_WITH_OP(MIN,  DO_OP_MIN)
ROCM_REDUCE_WITH_OP(SUM,  DO_OP_SUM)
ROCM_REDUCE_WITH_OP(PROD, DO_OP_PROD)
ROCM_REDUCE_WITH_OP(LAND, DO_OP_LAND)
ROCM_REDUCE_WITH_OP(BAND, DO_OP_BAND)
ROCM_REDUCE_WITH_OP(LOR,  DO_OP_LOR)
ROCM_REDUCE_WITH_OP(BOR,  DO_OP_BOR)
ROCM_REDUCE_WITH_OP(LXOR, DO_OP_LXOR)
ROCM_REDUCE_WITH_OP(BXOR, DO_OP_BXOR)

ROCM_REDUCE_WITH_OP_SPECIALIZED(MAX, DO_OP_MAX_HALF, __half)
ROCM_REDUCE_WITH_OP_SPECIALIZED(MIN, DO_OP_MIN_HALF, __half)
ROCM_REDUCE_WITH_OP_SPECIALIZED(SUM, DO_OP_SUM_HALF, __half)
ROCM_REDUCE_WITH_OP_SPECIALIZED(PROD, DO_OP_PROD_HALF, __half)

ROCM_REDUCE_WITH_OP_SPECIALIZED(SUM, DO_OP_SUM_FLOAT_COMPLEX, hipFloatComplex)
ROCM_REDUCE_WITH_OP_SPECIALIZED(PROD, DO_OP_PROD_FLOAT_COMPLEX, hipFloatComplex)

ROCM_REDUCE_WITH_OP_SPECIALIZED(SUM, DO_OP_SUM_DOUBLE_COMPLEX, hipDoubleComplex)
ROCM_REDUCE_WITH_OP_SPECIALIZED(PROD, DO_OP_PROD_DOUBLE_COMPLEX,
                                hipDoubleComplex)

ROCM_REDUCE_WITH_OP_SPECIALIZED(MAX, DO_OP_MAX_BFLOAT16, hip_bfloat16)
ROCM_REDUCE_WITH_OP_SPECIALIZED(MIN, DO_OP_MIN_BFLOAT16, hip_bfloat16)
ROCM_REDUCE_WITH_OP_SPECIALIZED(SUM, DO_OP_SUM_BFLOAT16, hip_bfloat16)
ROCM_REDUCE_WITH_OP_SPECIALIZED(PROD, DO_OP_PROD_BFLOAT16, hip_bfloat16)

#define LAUNCH_KERNEL(NAME, type, src1, src2, dest, count, s, b, t)            \
    do {                                                                       \
        UCC_REDUCE_ROCM_ ## NAME<type> <<<b, t, 0, s>>>(src1, src2,            \
                                                        dest, count);          \
    } while(0)


#define DT_REDUCE_INT(type, op, src1_p, src2_p, dest_p, count, s, b, t) do {   \
        const type *sbuf1 = (type *)src1_p;                                    \
        const type *sbuf2= (type *)src2_p;                                     \
        type *dest = (type *)dest_p;                                           \
        switch(op) {                                                           \
        case UCC_OP_MAX:                                                       \
            LAUNCH_KERNEL(MAX, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            LAUNCH_KERNEL(MIN, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_SUM:                                                       \
            LAUNCH_KERNEL(SUM, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL(PROD, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            LAUNCH_KERNEL(LAND, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            LAUNCH_KERNEL(BAND, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            LAUNCH_KERNEL(LOR, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            LAUNCH_KERNEL(BOR, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            LAUNCH_KERNEL(LXOR, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            LAUNCH_KERNEL(BXOR, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_rocm.super, "int dtype does not support "         \
                     "requested reduce op: %s", ucc_reduction_op_str(op));     \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while(0)

#define DT_REDUCE_FLOAT(type, op, src1_p, src2_p, dest_p, count, s, b, t)      \
    do {                                                                       \
        const type *sbuf1 = (const type *)src1_p;                              \
        const type *sbuf2 = (const type *)src2_p;                              \
        type       *dest  = (type *)dest_p;                                    \
        switch (op) {                                                          \
        case UCC_OP_MAX:                                                       \
            LAUNCH_KERNEL(MAX, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            LAUNCH_KERNEL(MIN, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_SUM:                                                       \
        case UCC_OP_AVG:                                                       \
            LAUNCH_KERNEL(SUM, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL(PROD, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_rocm.super,                                       \
                     "float dtype does not support "                           \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(op));                                \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DT_REDUCE_FLOAT_COMPLEX(type, op, src1_p, src2_p, dest_p, count, s, b, \
                                t)                                             \
    do {                                                                       \
        const type *sbuf1 = (const type *)src1_p;                              \
        const type *sbuf2 = (const type *)src2_p;                              \
        type *      dest  = (type *)dest_p;                                    \
        switch (op) {                                                          \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            LAUNCH_KERNEL(SUM, type, sbuf1, sbuf2, dest, count, s, b, t);      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            LAUNCH_KERNEL(PROD, type, sbuf1, sbuf2, dest, count, s, b, t);     \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_rocm.super,                                       \
                     "float complex dtype does not support "                   \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(op));                                \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_rocm_reduce(const void *src1, const void *src2, void *dst,
                                size_t count, ucc_datatype_t dt,
                                ucc_reduction_op_t op)
{
    int           th     = MC_ROCM_CONFIG->reduce_num_threads;
    unsigned long bk     = (count + th - 1)/th;
    hipStream_t  stream;

    UCC_MC_ROCM_INIT_STREAM();
    stream = ucc_mc_rocm.stream;
    if (MC_ROCM_CONFIG->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        bk = ucc_min(bk, MC_ROCM_CONFIG->reduce_num_blocks);
    }
    switch (dt)
    {
        case UCC_DT_INT16:
            DT_REDUCE_INT(int16_t, op, src1, src2, dst, count, stream, bk, th);
            break;
        case UCC_DT_INT32:
            DT_REDUCE_INT(int32_t, op, src1, src2, dst, count, stream, bk, th);
            break;
        case UCC_DT_INT64:
            DT_REDUCE_INT(int64_t, op, src1, src2, dst, count, stream, bk, th);
            break;
        case UCC_DT_FLOAT16:
            ucc_assert(2 == sizeof(__half));
            DT_REDUCE_FLOAT(__half, op, src1, src2, dst, count, stream, bk, th);
            break;
        case UCC_DT_FLOAT32:
            ucc_assert(4 == sizeof(float));
            DT_REDUCE_FLOAT(float, op, src1, src2, dst, count, stream, bk, th);
            break;
        case UCC_DT_FLOAT64:
            ucc_assert(8 == sizeof(double));
            DT_REDUCE_FLOAT(double, op, src1, src2, dst, count, stream, bk, th);
            break;
        case UCC_DT_FLOAT32_COMPLEX:
            DT_REDUCE_FLOAT_COMPLEX(hipFloatComplex, op, src1, src2, dst, count,
                                    stream, bk, th);
	    break;
        case UCC_DT_FLOAT64_COMPLEX:
            DT_REDUCE_FLOAT_COMPLEX(hipDoubleComplex, op, src1, src2, dst, count,
                                    stream, bk, th);
	    break;
        case UCC_DT_BFLOAT16:
            ucc_assert(2 == sizeof(hip_bfloat16));
            DT_REDUCE_FLOAT(hip_bfloat16, op, src1, src2, dst, count, stream,
                            bk, th);
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
