/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "utils/ucc_math_op.h"
#include "ec_cpu.h"
#include <complex.h>

#define DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, OP)                        \
    do {                                                                       \
        size_t _i, _j;                                                         \
        switch (_n_srcs) {                                                     \
        case 2:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_2(s[0][_i], s[1][_i]);                            \
            }                                                                  \
            break;                                                             \
        case 3:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_3(s[0][_i], s[1][_i], s[2][_i]);                  \
            }                                                                  \
            break;                                                             \
        case 4:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_4(s[0][_i], s[1][_i], s[2][_i], s[3][_i]);        \
            }                                                                  \
            break;                                                             \
        case 5:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] =                                                        \
                    OP##_5(s[0][_i], s[1][_i], s[2][_i], s[3][_i], s[4][_i]);  \
            }                                                                  \
            break;                                                             \
        case 6:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_6(s[0][_i], s[1][_i], s[2][_i], s[3][_i],         \
                               s[4][_i], s[5][_i]);                            \
            }                                                                  \
            break;                                                             \
        case 7:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_7(s[0][_i], s[1][_i], s[2][_i], s[3][_i],         \
                               s[4][_i], s[5][_i], s[6][_i]);                  \
            }                                                                  \
            break;                                                             \
        case 8:                                                                \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_8(s[0][_i], s[1][_i], s[2][_i], s[3][_i],         \
                               s[4][_i], s[5][_i], s[6][_i], s[7][_i]);        \
            }                                                                  \
            break;                                                             \
        default:                                                               \
            for (_i = 0; _i < _count; _i++) {                                  \
                d[_i] = OP##_8(s[0][_i], s[1][_i], s[2][_i], s[3][_i],         \
                               s[4][_i], s[5][_i], s[6][_i], s[7][_i]);        \
            }                                                                  \
            for (_j = 8; _j < _n_srcs; _j++) {                                 \
                for (_i = 0; _i < _count; _i++) {                              \
                    d[_i] = OP##_2(d[_i], s[_j][_i]);                          \
                }                                                              \
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
        const type **restrict s = (const type **restrict)_srcs;                \
        type *restrict        d = (type * restrict) _dst;                      \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_SUM);            \
            if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                 \
                VEC_OP(d, _count, task->alpha);                                \
            }                                                                  \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_MIN);            \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_MAX);            \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_PROD);           \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_LAND);           \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_BAND);           \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_LOR);            \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_BOR);            \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_LXOR);           \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_BXOR);           \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_cpu.super,                                        \
                     "float dtype does not support "                           \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_WITH_OP_BFLOAT16(_srcs, _dst, _count, _n_srcs, _OP,       \
                                      _alpha)                                  \
    do {                                                                       \
        float     _tmp;                                                        \
        size_t    _i, _j;                                                      \
        int16_t **_s = (int16_t **)_srcs;                                      \
        int16_t * _d = (int16_t *)_dst;                                        \
        for (_i = 0; _i < _count; _i++) {                                      \
            _tmp = _OP(bfloat16tofloat32(&_s[0][_i]),                          \
                       bfloat16tofloat32(&_s[1][_i]));                         \
            for (_j = 2; _j < _n_srcs; _j++) {                                 \
                _tmp = _OP(_tmp, bfloat16tofloat32(&_s[_j][_i]));              \
            }                                                                  \
            float32tobfloat16(_tmp *_alpha, &_d[_i]);                          \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_BFLOAT16(_srcs, _dst, _op, _count, _n_srcs)               \
    do {                                                                       \
        float _a = (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) ? task->alpha \
                                                                 : 1.0f;       \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(_srcs, _dst, _count, _n_srcs,        \
                                          DO_OP_SUM, _a);                      \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(_srcs, _dst, _count, _n_srcs,        \
                                          DO_OP_PROD, _a);                     \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(_srcs, _dst, _count, _n_srcs,        \
                                          DO_OP_MIN, _a);                      \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP_BFLOAT16(_srcs, _dst, _count, _n_srcs,        \
                                          DO_OP_MAX, _a);                      \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_cpu.super,                                        \
                     "bfloat16 dtype does not support "                        \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_FLOAT(type, _srcs, _dst, _op, _count, _n_srcs)            \
    do {                                                                       \
        const type **restrict s = (const type **restrict)_srcs;                \
        type *restrict        d = (type * restrict) _dst;                      \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_SUM);            \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_PROD);           \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_MIN);            \
            break;                                                             \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_MAX);            \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_cpu.super,                                        \
                     "float dtype does not support "                           \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                     \
            VEC_OP(d, _count, task->alpha);                                    \
        }                                                                      \
    } while (0)

#define DO_DT_REDUCE_FLOAT_COMPLEX(type, _srcs, _dst, _op, _count, _n_srcs)    \
    do {                                                                       \
        const type **restrict s = (const type **restrict)_srcs;                \
        type *restrict        d = (type * restrict) _dst;                      \
        switch (_op) {                                                         \
        case UCC_OP_AVG:                                                       \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_SUM);            \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s, d, _count, _n_srcs, DO_OP_PROD);           \
            break;                                                             \
        default:                                                               \
            ec_error(&ucc_ec_cpu.super,                                        \
                     "float complex dtype does not support "                   \
                     "requested reduce op: %s",                                \
                     ucc_reduction_op_str(_op));                               \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                     \
            VEC_OP(d, _count, task->alpha);          \
        }                                                                      \
    } while (0)

ucc_status_t ucc_ec_cpu_reduce(ucc_eee_task_reduce_t *task, uint16_t flags)
{
    void **srcs = (flags & UCC_EEE_TASK_FLAG_REDUCE_SRCS_EXT) ? task->srcs_ext
                                                              : task->srcs;

    switch (task->dt) {
    case UCC_DT_INT8:
        DO_DT_REDUCE_INT(int8_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_INT16:
        DO_DT_REDUCE_INT(int16_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_INT32:
        DO_DT_REDUCE_INT(int32_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_INT64:
        DO_DT_REDUCE_INT(int64_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_UINT8:
        DO_DT_REDUCE_INT(uint8_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_UINT16:
        DO_DT_REDUCE_INT(uint16_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_UINT32:
        DO_DT_REDUCE_INT(uint32_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_UINT64:
        DO_DT_REDUCE_INT(uint64_t, srcs, task->dst, task->op, task->count,
                         task->n_srcs);
        break;
    case UCC_DT_FLOAT32:
#if SIZEOF_FLOAT == 4
        DO_DT_REDUCE_FLOAT(float, srcs, task->dst, task->op, task->count,
                           task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64:
#if SIZEOF_DOUBLE == 8
        DO_DT_REDUCE_FLOAT(double, srcs, task->dst, task->op, task->count,
                           task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT128:
#if SIZEOF_LONG_DOUBLE == 16
        DO_DT_REDUCE_FLOAT(long double, srcs, task->dst, task->op, task->count,
                           task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_BFLOAT16:
        DO_DT_REDUCE_BFLOAT16(srcs, task->dst, task->op, task->count,
                              task->n_srcs);
        break;
    case UCC_DT_FLOAT32_COMPLEX:
#if SIZEOF_FLOAT__COMPLEX == 8
        DO_DT_REDUCE_FLOAT_COMPLEX(float complex, srcs, task->dst, task->op,
                                   task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64_COMPLEX:
#if SIZEOF_DOUBLE__COMPLEX == 16
        DO_DT_REDUCE_FLOAT_COMPLEX(double complex, srcs, task->dst, task->op,
                                   task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT128_COMPLEX:
#if SIZEOF_LONG_DOUBLE__COMPLEX == 32
        DO_DT_REDUCE_FLOAT_COMPLEX(long double complex, srcs, task->dst,
                                   task->op, task->count, task->n_srcs);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    default:
        ec_error(&ucc_ec_cpu.super, "unsupported reduction type (%s)",
                 ucc_datatype_str(task->dt));
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}
