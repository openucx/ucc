/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cpu.h"
#include "utils/ucc_malloc.h"

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
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cpu.super.config_table,
                                &ucc_config_global_list);
