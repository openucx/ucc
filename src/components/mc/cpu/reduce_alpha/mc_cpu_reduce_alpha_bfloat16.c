/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "mc_cpu.h"
#include "reduce/mc_cpu_reduce.h"

#undef VEC_OP
#define VEC_OP(_d, OP)                                                         \
    do {                                                                       \
        size_t _i;                                                             \
        for (_i = 0; _i < count; _i++) {                                       \
            float32tobfloat16(OP(bfloat16tofloat32(&_d[_i]), alpha),           \
                              &_d[_i]);                                        \
        }                                                                      \
    } while (0)

ucc_status_t
ucc_mc_cpu_reduce_multi_alpha_bfloat16(const void *src1, const void *src2,
                                       void *dst, size_t n_vectors, size_t count,
                                       size_t stride, ucc_reduction_op_t reduce_op,
                                       ucc_reduction_op_t vector_op, float alpha)
{
    ucc_status_t status;
    status = ucc_mc_cpu_reduce_multi_bfloat16(src1, src2, dst, n_vectors, count,
                                              stride, reduce_op);
    if (ucc_unlikely(status != UCC_OK)){
        return status;
    }
    DO_VEC_OP(uint16_t, dst);
    return UCC_OK;
}
