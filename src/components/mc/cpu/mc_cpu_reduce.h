/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CPU_REDUCE_H_
#define UCC_MC_CPU_REDUCE_H_

#define  DO_OP_MAX(_v1, _v2) (_v1 > _v2 ? _v1 : _v2)
#define  DO_OP_MIN(_v1, _v2) (_v1 < _v2 ? _v1 : _v2)
#define  DO_OP_SUM(_v1, _v2) (_v1 + _v2)
#define DO_OP_PROD(_v1, _v2) (_v1 * _v2)
#define DO_OP_LAND(_v1, _v2) (_v1 && _v2)
#define DO_OP_BAND(_v1, _v2) (_v1 & _v2)
#define  DO_OP_LOR(_v1, _v2) (_v1 || _v2)
#define  DO_OP_BOR(_v1, _v2) (_v1 | _v2)
#define DO_OP_LXOR(_v1, _v2) ((!_v1) != (!_v2))
#define DO_OP_BXOR(_v1, _v2) (_v1 ^ _v2)

#define DO_DT_REDUCE_WITH_OP(s1, s2, d, count, OP) do {                        \
        size_t i;                                                              \
        for (i=0; i<count; i++) {                                              \
            d[i] = OP(s1[i], s2[i]);                                           \
        }                                                                      \
    } while(0)

#define DO_DT_REDUCE_INT(type, op, src1_p, src2_p, dest_p, count) do {         \
        const type *s1 = (const type *)src1_p;                                 \
        const type *s2 = (const type *)src2_p;                                 \
        type *d = (type *)dest_p;                                              \
        switch(op) {                                                           \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_MAX);                 \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_MIN);                 \
            break;                                                             \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_SUM);                 \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_PROD);                \
            break;                                                             \
        case UCC_OP_LAND:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_LAND);                \
            break;                                                             \
        case UCC_OP_BAND:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_BAND);                \
            break;                                                             \
        case UCC_OP_LOR:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_LOR);                 \
            break;                                                             \
        case UCC_OP_BOR:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_BOR);                 \
            break;                                                             \
        case UCC_OP_LXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_LXOR);                \
            break;                                                             \
        case UCC_OP_BXOR:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_BXOR);                \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_cpu.super, "int dtype does not support "          \
                                        "requested reduce op: %d", op);        \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while(0)

#define DO_DT_REDUCE_FLOAT(type, op, src1_p, src2_p, dest_p, count) do {       \
        const type *s1 = (const type *)src1_p;                                 \
        const type *s2 = (const type *)src2_p;                                 \
        type *d = (type *)dest_p;                                              \
        switch(op) {                                                           \
        case UCC_OP_MAX:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_MAX);                 \
            break;                                                             \
        case UCC_OP_MIN:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_MIN);                 \
            break;                                                             \
        case UCC_OP_SUM:                                                       \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_SUM);                 \
            break;                                                             \
        case UCC_OP_PROD:                                                      \
            DO_DT_REDUCE_WITH_OP(s1, s2, d, count, DO_OP_PROD);                \
            break;                                                             \
        default:                                                               \
            mc_error(&ucc_mc_cpu.super, "float dtype does not support "        \
                                        "requested reduce op: %d", op);        \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while(0)

#endif
