/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cpu.h"
#include "reduce/mc_cpu_reduce.h"
#include "utils/ucc_malloc.h"
#include <sys/types.h>

static ucc_config_field_t ucc_mc_cpu_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cpu_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {NULL}};

static ucc_status_t ucc_mc_cpu_init()
{
    ucc_strncpy_safe(ucc_mc_cpu.super.config->log_component.name,
                     ucc_mc_cpu.super.super.name,
                     sizeof(ucc_mc_cpu.super.config->log_component.name));
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_finalize()
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_mem_alloc(void **ptr, size_t size)
{
    (*ptr) = ucc_malloc(size, "mc cpu");
    if (ucc_unlikely(!(*ptr))) {
        mc_error(&ucc_mc_cpu.super, "failed to allocate %zd bytes", size);
        return UCC_ERR_NO_MEMORY;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_reduce_multi(const void *src1, const void *src2,
                                            void *dst, size_t size,
                                            size_t count, size_t stride,
                                            ucc_datatype_t     dt,
                                            ucc_reduction_op_t op)
{
    switch(dt) {
    case UCC_DT_INT8:
        return ucc_mc_cpu_reduce_multi_int8(src1, src2, dst, size, count,
                                            stride, dt, op);
    case UCC_DT_INT16:
        return ucc_mc_cpu_reduce_multi_int16(src1, src2, dst, size, count,
                                             stride, dt, op);
    case UCC_DT_INT32:
        return ucc_mc_cpu_reduce_multi_int32(src1, src2, dst, size, count,
                                             stride, dt, op);
    case UCC_DT_INT64:
        return ucc_mc_cpu_reduce_multi_int64(src1, src2, dst, size, count,
                                             stride, dt, op);
    case UCC_DT_UINT8:
        return ucc_mc_cpu_reduce_multi_uint8(src1, src2, dst, size, count,
                                             stride, dt, op);
    case UCC_DT_UINT16:
        return ucc_mc_cpu_reduce_multi_uint16(src1, src2, dst, size, count,
                                              stride, dt, op);
    case UCC_DT_UINT32:
        return ucc_mc_cpu_reduce_multi_uint32(src1, src2, dst, size, count,
                                              stride, dt, op);
    case UCC_DT_UINT64:
        return ucc_mc_cpu_reduce_multi_uint64(src1, src2, dst, size, count,
                                              stride, dt, op);
    case UCC_DT_FLOAT32:
        ucc_assert(4 == sizeof(float));
        return ucc_mc_cpu_reduce_multi_float(src1, src2, dst, size, count,
                                             stride, dt, op);
    case UCC_DT_FLOAT64:
        ucc_assert(8 == sizeof(double));
        return ucc_mc_cpu_reduce_multi_double(src1, src2, dst, size, count,
                                              stride, dt, op);
    default:
        mc_error(&ucc_mc_cpu.super, "unsupported reduction type (%d)", dt);
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_reduce(const void *src1, const void *src2,
                                      void *dst, size_t count,
                                      ucc_datatype_t dt, ucc_reduction_op_t op)
{
    return ucc_mc_cpu_reduce_multi(src1, src2, dst, 1, count, 0, dt, op);
}

static ucc_status_t ucc_mc_cpu_mem_free(void *ptr)
{
    ucc_free(ptr);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_memcpy(void *dst, const void *src, size_t len,
                                      ucc_memory_type_t dst_mem, //NOLINT
                                      ucc_memory_type_t src_mem) //NOLINT
{
    ucc_assert((dst_mem == UCC_MEMORY_TYPE_HOST) &&
               (src_mem == UCC_MEMORY_TYPE_HOST));
    memcpy(dst, src, len);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_mem_query(const void *ptr, //NOLINT
                                         ucc_mem_attr_t *mem_attr) //NOLINT
{
    /* not supposed to be used */
    mc_error(&ucc_mc_cpu.super, "host memory component shouldn't be used for"
                                "mem type detection");
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_ee_cpu_task_post(void *ee_context, //NOLINT
                                  void **ee_req)
{
    *ee_req = NULL;
    return UCC_OK;
}

ucc_status_t ucc_ee_cpu_task_query(void *ee_req) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_ee_cpu_task_end(void *ee_req) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_ee_cpu_create_event(void **event)
{
    *event = NULL;
    return UCC_OK;
}

ucc_status_t ucc_ee_cpu_destroy_event(void *event) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_ee_cpu_event_post(void *ee_context, //NOLINT
                                   void *event) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_ee_cpu_event_test(void *event) //NOLINT
{
    return UCC_OK;
}


ucc_mc_cpu_t ucc_mc_cpu = {
    .super.super.name       = "cpu mc",
    .super.ref_cnt          = 0,
    .super.type             = UCC_MEMORY_TYPE_HOST,
    .super.ee_type          = UCC_EE_CPU_THREAD,
    .super.init             = ucc_mc_cpu_init,
    .super.finalize         = ucc_mc_cpu_finalize,
    .super.ops.mem_query    = ucc_mc_cpu_mem_query,
    .super.ops.mem_alloc    = ucc_mc_cpu_mem_alloc,
    .super.ops.mem_free     = ucc_mc_cpu_mem_free,
    .super.ops.reduce       = ucc_mc_cpu_reduce,
    .super.ops.reduce_multi = ucc_mc_cpu_reduce_multi,
    .super.ops.memcpy       = ucc_mc_cpu_memcpy,
    .super.config_table     =
        {
            .name   = "CPU memory component",
            .prefix = "MC_CPU_",
            .table  = ucc_mc_cpu_config_table,
            .size   = sizeof(ucc_mc_cpu_config_t),
        },
    .super.ee_ops.ee_task_post     = ucc_ee_cpu_task_post,
    .super.ee_ops.ee_task_query    = ucc_ee_cpu_task_query,
    .super.ee_ops.ee_task_end      = ucc_ee_cpu_task_end,
    .super.ee_ops.ee_create_event  = ucc_ee_cpu_create_event,
    .super.ee_ops.ee_destroy_event = ucc_ee_cpu_destroy_event,
    .super.ee_ops.ee_event_post    = ucc_ee_cpu_event_post,
    .super.ee_ops.ee_event_test    = ucc_ee_cpu_event_test,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cpu.super.config_table,
                                &ucc_config_global_list);
