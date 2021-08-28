/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "mc_cpu.h"
#include "reduce/mc_cpu_reduce.h"

ucc_status_t ucc_mc_cpu_reduce_multi_double(const void *src1, const void *src2,
                                            void *dst, size_t n_vectors,
                                            size_t count, size_t stride,
                                            ucc_reduction_op_t op)
{
    DO_DT_REDUCE_FLOAT(double, op, src1, src2, dst, n_vectors, count, stride);
    return UCC_OK;
}
