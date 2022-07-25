/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CPU_REDUCE_H_
#define UCC_MC_CPU_REDUCE_H_

#include "utils/ucc_math.h"
#include <complex.h>

#define OP_1(_s1, _s2, _i, _sc, _OP) _OP(_s1[_i], _s2[_i])
#define OP_2(_s1, _s2, _i, _sc, _OP)                                           \
    _OP((OP_1(_s1, _s2, _i, _sc, _OP)), _s2[_i + 1 * _sc])
#define OP_3(_s1, _s2, _i, _sc, _OP)                                           \
    _OP((OP_2(_s1, _s2, _i, _sc, _OP)), _s2[_i + 2 * _sc])
#define OP_4(_s1, _s2, _i, _sc, _OP)                                           \
    _OP((OP_3(_s1, _s2, _i, _sc, _OP)), _s2[_i + 3 * _sc])
#define OP_5(_s1, _s2, _i, _sc, _OP)                                           \
    _OP((OP_4(_s1, _s2, _i, _sc, _OP)), _s2[_i + 4 * _sc])
#define OP_6(_s1, _s2, _i, _sc, _OP)                                           \
    _OP((OP_5(_s1, _s2, _i, _sc, _OP)), _s2[_i + 5 * _sc])
#define OP_7(_s1, _s2, _i, _sc, _OP)                                           \
    _OP((OP_6(_s1, _s2, _i, _sc, _OP)), _s2[_i + 6 * _sc])

#define OP_N(_d, _s1, _s2, _sc, _OP, _n)                                       \
    do {                                                                       \
        size_t _i;                                                             \
        for (_i = 0; _i < count; _i++) {                                       \
            _d[_i] = OP_##_n(_s1, _s2, _i, _sc, _OP);                          \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, OP)               \
    do {                                                                       \
        size_t i;                                                              \
        size_t stride_count = stride / sizeof(s1[0]);                          \
        ucc_assert((stride % sizeof(s1[0])) == 0);                             \
        switch (size) {                                                        \
        case 1:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 1);                              \
            break;                                                             \
        case 2:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 2);                              \
            break;                                                             \
        case 3:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 3);                              \
            break;                                                             \
        case 4:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 4);                              \
            break;                                                             \
        case 5:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 5);                              \
            break;                                                             \
        case 6:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 6);                              \
            break;                                                             \
        case 7:                                                                \
            OP_N(d, s1, s2, stride_count, OP, 7);                              \
            break;                                                             \
        default:                                                               \
            OP_N(d, s1, s2, stride_count, OP, 3);                              \
            for (i = 1; i < size / 3; i++) {                                   \
                OP_N(d, d, (&s2[3 * i * stride_count]), stride_count, OP, 3);  \
            }                                                                  \
            if ((size % 3) == 2) {                                             \
                OP_N(d, d, (&s2[(size - 2) * stride_count]), stride_count, OP, \
                     2);                                                       \
            }                                                                  \
            else if ((size % 3) == 1) {                                        \
                OP_N(d, d, (&s2[(size - 1) * stride_count]), stride_count, OP, \
                     1);                                                       \
            }                                                                  \
            break;                                                             \
        }                                                                      \
    } while (0)

/* Note the use of "restrict" keyword in the macro below
   (same applies to DO_DT_REDUCE_FLOAT). It is critical for performance
   (depending on compiler). We use "restrict" here inspite of the fact that
   memory pointed by s2 and by d can overlap. This could lead to the correctness
   issue in generic case. However, in our case we only allow d to be equal
   to s2 when these two memories overlap. This guarantees that d[j] memory
   location can only be written after it was read. So there is no RAW violation
   here. We trick the compiler that way to produce optimized code. */
#define DO_DT_REDUCE_INT(type, op, src1_p, src2_p, dest_p, size, count,        \
                         stride)                                               \
    do {                                                                       \
        const type *restrict s1 = (const type *restrict)src1_p;                \
        const type *restrict s2 = (const type *restrict)src2_p;                \
        type *restrict       d  = (type * restrict) dest_p;                    \
        ucc_assert((ptrdiff_t)d <= (ptrdiff_t)src2_p ||                        \
                   (ptrdiff_t)d > (ptrdiff_t)src2_p + (size - 1) * stride +    \
                                      count * sizeof(type));                   \
        switch (op) {                                                          \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_MAX);   \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_MIN);   \
            break;                                                             \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_SUM);   \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_PROD);  \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_LAND);  \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_BAND);  \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_LOR);   \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_BOR);   \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_LXOR);  \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_BXOR);  \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_cpu.super,                                        \
                     "int dtype does not support "                             \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(op));                                \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_FLOAT(type, reduce_op, src1_p, src2_p, dest_p, size,      \
                           count, stride)                                      \
    do {                                                                       \
        const type *restrict s1 = (const type *restrict)src1_p;                \
        const type *restrict s2 = (const type *restrict)src2_p;                \
        type *restrict       d  = (type * restrict) dest_p;                    \
        ucc_assert((ptrdiff_t)d <= (ptrdiff_t)src2_p ||                        \
                   (ptrdiff_t)d > (ptrdiff_t)src2_p + (size - 1) * stride +    \
                                      count * sizeof(type));                   \
        switch (reduce_op) {                                                   \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_MAX);   \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_MIN);   \
            break;                                                             \
        case UCC_OP_SUM:                                                       \
        case UCC_OP_AVG:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_SUM);   \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_PROD);  \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_cpu.super,                                        \
                     "float dtype does not support "                           \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(reduce_op));                         \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_FLOAT_COMPLEX(type, reduce_op, src1_p, src2_p, dest_p,    \
                                   size, count, stride)                        \
    do {                                                                       \
        const type *restrict s1 = (const type *restrict)src1_p;                \
        const type *restrict s2 = (const type *restrict)src2_p;                \
        type *restrict       d  = (type * restrict) dest_p;                    \
        ucc_assert((ptrdiff_t)d <= (ptrdiff_t)src2_p ||                        \
                   (ptrdiff_t)d > (ptrdiff_t)src2_p + (size - 1) * stride +    \
                                      count * sizeof(type));                   \
        switch (reduce_op) {                                                   \
        case UCC_OP_SUM:                                                       \
        case UCC_OP_AVG:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_SUM);   \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, size, count, stride, DO_OP_PROD);  \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_cpu.super,                                        \
                     "float complex dtype does not support "                   \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(reduce_op));                         \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define VEC_OP(_d, OP)                                                         \
    do {                                                                       \
        size_t _i;                                                             \
        for (_i = 0; _i < count; _i++) {                                       \
            _d[_i] = OP(_d[_i], alpha);                                        \
        }                                                                      \
    } while (0)

#define DO_VEC_OP(type, _d)                                                    \
    do {                                                                       \
        type *restrict d  = (type * restrict) _d;                              \
        switch (vector_op) {                                                   \
        case UCC_OP_PROD:                                                      \
            VEC_OP(d, DO_OP_PROD);                                             \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_cpu.super,                                        \
                     "reduce multi with alpha does not support "               \
                     "requested vector op: %s",                                \
                     ucc_reduction_op_str(vector_op));                         \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define REDUCE_FN_DECLARE(_type)                                               \
    ucc_status_t ucc_mc_cpu_reduce_multi_##_type(                              \
        const void *src1, const void *src2, void *dst, size_t n_vectors,       \
        size_t count, size_t stride, ucc_reduction_op_t op)
REDUCE_FN_DECLARE(int8);
REDUCE_FN_DECLARE(int16);
REDUCE_FN_DECLARE(int32);
REDUCE_FN_DECLARE(int64);
REDUCE_FN_DECLARE(uint8);
REDUCE_FN_DECLARE(uint16);
REDUCE_FN_DECLARE(uint32);
REDUCE_FN_DECLARE(uint64);
REDUCE_FN_DECLARE(float);
REDUCE_FN_DECLARE(double);
REDUCE_FN_DECLARE(long_double);
REDUCE_FN_DECLARE(bfloat16);
REDUCE_FN_DECLARE(float_complex);
REDUCE_FN_DECLARE(double_complex);
REDUCE_FN_DECLARE(long_double_complex);

#define REDUCE_ALPHA_FN_DECLARE(_type, alpha_dt)                               \
    ucc_status_t ucc_mc_cpu_reduce_multi_alpha_##_type(                        \
        const void *src1, const void *src2, void *dst, size_t n_vectors,       \
        size_t count, size_t stride, ucc_reduction_op_t reduce_op,             \
        ucc_reduction_op_t vector_op, alpha_dt alpha)
REDUCE_ALPHA_FN_DECLARE(float, float);
REDUCE_ALPHA_FN_DECLARE(double, double);
REDUCE_ALPHA_FN_DECLARE(long, long double);
REDUCE_ALPHA_FN_DECLARE(bfloat16, float);
REDUCE_ALPHA_FN_DECLARE(float_complex, float);
REDUCE_ALPHA_FN_DECLARE(double_complex, double);
REDUCE_ALPHA_FN_DECLARE(long_complex, long double);

#endif
