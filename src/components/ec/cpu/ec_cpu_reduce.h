/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CPU_REDUCE_H_
#define UCC_EC_CPU_REDUCE_H_

#include "utils/ucc_math_op.h"
#include "ec_cpu.h"
#include <complex.h>

#define DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, OP)                  \
    do {                                                                       \
        size_t _i, _j;                                                         \
        type   _tmp;                                                           \
        size_t __count = _count;                                               \
        switch (_n_srcs) {                                                     \
        case 2:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_2(s[0][_i], s[1][_i]);                            \
            }                                                                  \
            break;                                                             \
        case 3:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_3(s[0][_i], s[1][_i], s[2][_i]);                  \
            }                                                                  \
            break;                                                             \
        case 4:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_4(s[0][_i], s[1][_i], s[2][_i], s[3][_i]);        \
            }                                                                  \
            break;                                                             \
        case 5:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_5(                                                \
                    s[0][_i], s[1][_i], s[2][_i], s[3][_i], s[4][_i]);         \
            }                                                                  \
            break;                                                             \
        case 6:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_6(                                                \
                    s[0][_i],                                                  \
                    s[1][_i],                                                  \
                    s[2][_i],                                                  \
                    s[3][_i],                                                  \
                    s[4][_i],                                                  \
                    s[5][_i]);                                                 \
            }                                                                  \
            break;                                                             \
        case 7:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_7(                                                \
                    s[0][_i],                                                  \
                    s[1][_i],                                                  \
                    s[2][_i],                                                  \
                    s[3][_i],                                                  \
                    s[4][_i],                                                  \
                    s[5][_i],                                                  \
                    s[6][_i]);                                                 \
            }                                                                  \
            break;                                                             \
        case 8:                                                                \
            for (_i = 0; _i < __count; _i++) {                                 \
                d[_i] = OP##_8(                                                \
                    s[0][_i],                                                  \
                    s[1][_i],                                                  \
                    s[2][_i],                                                  \
                    s[3][_i],                                                  \
                    s[4][_i],                                                  \
                    s[5][_i],                                                  \
                    s[6][_i],                                                  \
                    s[7][_i]);                                                 \
            }                                                                  \
            break;                                                             \
        default:                                                               \
            for (_i = 0; _i < __count; _i++) {                                 \
                _tmp = OP##_8(                                                 \
                    s[0][_i],                                                  \
                    s[1][_i],                                                  \
                    s[2][_i],                                                  \
                    s[3][_i],                                                  \
                    s[4][_i],                                                  \
                    s[5][_i],                                                  \
                    s[6][_i],                                                  \
                    s[7][_i]);                                                 \
                for (_j = 8; _j < _n_srcs; _j++) {                             \
                    _tmp = OP##_2(_tmp, s[_j][_i]);                            \
                }                                                              \
                d[_i] = _tmp;                                                  \
            }                                                                  \
            break;                                                             \
        }                                                                      \
    } while (0)

#define VEC_OP(_d, _count, _alpha)                                             \
    do {                                                                       \
        size_t _i;                                                             \
        for (_i = 0; _i < _count; _i++) {                                      \
            _d[_i] = _d[_i] * _alpha;                                          \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_INT(type, _srcs, _dst, _op, _count, _n_srcs)              \
    do {                                                                       \
        const type **restrict s = (const type **)_srcs;                        \
        type *restrict d        = (type *)_dst;                                \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_SUM);      \
            if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                 \
                VEC_OP(d, _count, task->alpha);                                \
            }                                                                  \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_MIN);      \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_MAX);      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_PROD);     \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_LAND);     \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_BAND);     \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_LOR);      \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_BOR);      \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_LXOR);     \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_BXOR);     \
            break;                                                             \
        default:                                                               \
            ec_error(                                                          \
                &ucc_ec_cpu.super,                                             \
                "int dtype does not support "                                  \
                "requested reduce op: %s",                                     \
                ucc_reduction_op_str(_op));                                    \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_WITH_OP_BFLOAT16(                                         \
    _srcs, _dst, _count, _n_srcs, _OP, _alpha)                                 \
    do {                                                                       \
        float     _tmp;                                                        \
        size_t    _i, _j;                                                      \
        int16_t **_s = (int16_t **)_srcs;                                      \
        int16_t  *_d = (int16_t *)_dst;                                        \
        for (_i = 0; _i < _count; _i++) {                                      \
            _tmp = _OP(                                                        \
                bfloat16tofloat32(&_s[0][_i]), bfloat16tofloat32(&_s[1][_i])); \
            for (_j = 2; _j < _n_srcs; _j++) {                                 \
                _tmp = _OP(_tmp, bfloat16tofloat32(&_s[_j][_i]));              \
            }                                                                  \
            float32tobfloat16(_tmp * _alpha, &_d[_i]);                         \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_BFLOAT16(_srcs, _dst, _op, _count, _n_srcs)               \
    do {                                                                       \
        float _a = (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) ? task->alpha \
                                                                 : 1.0f;       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(                                     \
                _srcs, _dst, _count, _n_srcs, DO_OP_SUM, _a);                  \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(                                     \
                _srcs, _dst, _count, _n_srcs, DO_OP_PROD, _a);                 \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(                                     \
                _srcs, _dst, _count, _n_srcs, DO_OP_MIN, _a);                  \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(                                     \
                _srcs, _dst, _count, _n_srcs, DO_OP_MAX, _a);                  \
            break;                                                             \
        default:                                                               \
            ec_error(                                                          \
                &ucc_ec_cpu.super,                                             \
                "bfloat16 dtype does not support "                             \
                "requested reduce op: %s",                                     \
                ucc_reduction_op_str(_op));                                    \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_FLOAT(type, _srcs, _dst, _op, _count, _n_srcs)            \
    do {                                                                       \
        const type **restrict s = (const type **)_srcs;                        \
        type *restrict d        = (type *)_dst;                                \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_SUM);      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_PROD);     \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_MIN);      \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_MAX);      \
            break;                                                             \
        default:                                                               \
            ec_error(                                                          \
                &ucc_ec_cpu.super,                                             \
                "float dtype does not support "                                \
                "requested reduce op: %s",                                     \
                ucc_reduction_op_str(_op));                                    \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                     \
            VEC_OP(d, _count, task->alpha);                                    \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_FLOAT_COMPLEX(type, _srcs, _dst, _op, _count, _n_srcs)    \
    do {                                                                       \
        const type **restrict s = (const type **)_srcs;                        \
        type *restrict d        = (type *)_dst;                                \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_SUM);      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(type, s, d, _count, _n_srcs, DO_OP_PROD);     \
            break;                                                             \
        default:                                                               \
            ec_error(                                                          \
                &ucc_ec_cpu.super,                                             \
                "float complex dtype does not support "                        \
                "requested reduce op: %s",                                     \
                ucc_reduction_op_str(_op));                                    \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                     \
            VEC_OP(d, _count, task->alpha);                                    \
        }                                                                      \
    } while (0)

ucc_status_t ucc_ec_cpu_reduce_int8(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags);

ucc_status_t ucc_ec_cpu_reduce_int16(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags);

ucc_status_t ucc_ec_cpu_reduce_int32(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags);

ucc_status_t ucc_ec_cpu_reduce_int64(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags);

ucc_status_t ucc_ec_cpu_reduce_float(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags);

ucc_status_t ucc_ec_cpu_reduce_complex(
    ucc_eee_task_reduce_t *task, void *restrict dst, void *const *restrict srcs,
    uint16_t               flags);

#endif
