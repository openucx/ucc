/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "mc_cpu.h"
#include "reduce/mc_cpu_reduce.h"

ucc_status_t ucc_mc_cpu_reduce_multi_int8(const void *src1, const void *src2,
                                          void *dst, size_t size, size_t count,
                                          size_t stride, ucc_datatype_t dt,
                                          ucc_reduction_op_t op)
{
    DO_DT_REDUCE_INT(int8_t, op, src1, src2, dst, size, count, stride);
    return UCC_OK;
}
