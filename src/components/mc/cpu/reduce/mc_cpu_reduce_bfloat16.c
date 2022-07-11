/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "mc_cpu.h"
#include "reduce/mc_cpu_reduce.h"

#define OP_BFLOAT16_1(_s1, _s2, _i, _sc, _OP)                                  \
    _OP(bfloat16tofloat32(&_s1[_i]), bfloat16tofloat32(&_s2[_i]))
#define OP_BFLOAT16_2(_s1, _s2, _i, _sc, _OP)                                  \
    _OP((OP_BFLOAT16_1(_s1, _s2, _i, _sc, _OP)),                               \
        bfloat16tofloat32(&_s2[_i + 1 * _sc]))
#define OP_BFLOAT16_3(_s1, _s2, _i, _sc, _OP)                                  \
    _OP((OP_BFLOAT16_2(_s1, _s2, _i, _sc, _OP)),                               \
        bfloat16tofloat32(&_s2[_i + 2 * _sc]))
#define OP_BFLOAT16_4(_s1, _s2, _i, _sc, _OP)                                  \
    _OP((OP_BFLOAT16_3(_s1, _s2, _i, _sc, _OP)),                               \
        bfloat16tofloat32(&_s2[_i + 3 * _sc]))
#define OP_BFLOAT16_5(_s1, _s2, _i, _sc, _OP)                                  \
    _OP((OP_BFLOAT16_4(_s1, _s2, _i, _sc, _OP)),                               \
        bfloat16tofloat32(&_s2[_i + 4 * _sc]))
#define OP_BFLOAT16_6(_s1, _s2, _i, _sc, _OP)                                  \
    _OP((OP_BFLOAT16_5(_s1, _s2, _i, _sc, _OP)),                               \
        bfloat16tofloat32(&_s2[_i + 5 * _sc]))
#define OP_BFLOAT16_7(_s1, _s2, _i, _sc, _OP)                                  \
    _OP((OP_BFLOAT16_6(_s1, _s2, _i, _sc, _OP)),                               \
        bfloat16tofloat32(&_s2[_i + 6 * _sc]))

#undef OP_N
#define OP_N(_d, _s1, _s2, _sc, _OP, _n)                                       \
    do {                                                                       \
        size_t _i;                                                             \
        for (_i = 0; _i < count; _i++) {                                       \
            float32tobfloat16(OP_BFLOAT16_##_n(_s1, _s2, _i, _sc, _OP),        \
                              &_d[_i]);                                        \
        }                                                                      \
    } while (0)

ucc_status_t ucc_mc_cpu_reduce_multi_bfloat16(const void *src1,
                                              const void *src2, void *dst,
                                              size_t n_vectors, size_t count,
                                              size_t             stride,
                                              ucc_reduction_op_t op)
{
    DO_DT_REDUCE_FLOAT(uint16_t, op, src1, src2, dst, n_vectors,
                       count, stride);
    return UCC_OK;
}
