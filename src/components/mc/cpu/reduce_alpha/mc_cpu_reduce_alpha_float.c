/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "mc_cpu.h"
#include "reduce/mc_cpu_reduce.h"

ucc_status_t
ucc_mc_cpu_reduce_multi_alpha_float(const void *src1, const void *src2,
                                    void *dst, size_t n_vectors, size_t count,
                                    size_t stride, ucc_reduction_op_t reduce_op,
                                    ucc_reduction_op_t vector_op, float alpha)
{
    DO_DT_REDUCE_FLOAT(float, reduce_op, src1, src2, dst, n_vectors, count,
                       stride);
    DO_VEC_OP(float, dst);
    return UCC_OK;
}
