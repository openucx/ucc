/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cpu.h"
#include "mc_cpu_reduce.h"
#include "utils/ucc_malloc.h"
#include <sys/types.h>

static ucc_config_field_t ucc_mc_cpu_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cpu_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {NULL}};

static ucc_status_t ucc_mc_cpu_init()
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_finalize()
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_mem_alloc(void **ptr, size_t size)
{
    (*ptr) = ucc_malloc(size, "mc cpu");
    if (!(*ptr)) {
        mc_error(&ucc_mc_cpu.super, "failed to allocate %zd bytes", size);
        return UCC_ERR_NO_MEMORY;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_reduce(const void *src1, const void *src2,
                                      void *dst, size_t count,
                                      ucc_datatype_t dt, ucc_reduction_op_t op)
{
    switch(dt) {
    case UCC_DT_INT16:
        DO_DT_REDUCE_INT(int16_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_INT32:
        DO_DT_REDUCE_INT(int32_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_INT64:
        DO_DT_REDUCE_INT(int64_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_FLOAT32:
        ucc_assert(4 == sizeof(float));
        DO_DT_REDUCE_FLOAT(float, op, src1, src2, dst, count);
        break;
    case UCC_DT_FLOAT64:
        ucc_assert(8 == sizeof(double));
        DO_DT_REDUCE_FLOAT(double, op, src1, src2, dst, count);
        break;
    default:
        mc_error(&ucc_mc_cpu.super, "unsupported reduction type (%d)", dt);
        return UCC_ERR_NOT_SUPPORTED;
    }
    return 0;
}

static ucc_status_t ucc_mc_cpu_mem_free(void *ptr)
{
    ucc_free(ptr);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_mem_type(const void *ptr,
                                        ucc_memory_type_t *mem_type)
{
    /* not supposed to be used */
    mc_error(&ucc_mc_cpu.super, "host memory component shouldn't be used for"
                                "mem type detection");
    return UCC_ERR_NOT_SUPPORTED;
}


ucc_mc_cpu_t ucc_mc_cpu = {
    .super.super.name = "cpu mc",
    .super.ref_cnt    = 0,
    .super.type       = UCC_MEMORY_TYPE_HOST,
    .super.config_table =
        {
            .name   = "CPU memory component",
            .prefix = "MC_CPU_",
            .table  = ucc_mc_cpu_config_table,
            .size   = sizeof(ucc_mc_cpu_config_t),
        },
    .super.init          = ucc_mc_cpu_init,
    .super.finalize      = ucc_mc_cpu_finalize,
    .super.ops.mem_type  = ucc_mc_cpu_mem_type,
    .super.ops.mem_alloc = ucc_mc_cpu_mem_alloc,
    .super.ops.mem_free  = ucc_mc_cpu_mem_free,
    .super.ops.reduce    = ucc_mc_cpu_reduce,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cpu.super.config_table,
                                &ucc_config_global_list);
