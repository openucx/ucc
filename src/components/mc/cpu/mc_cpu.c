/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cpu.h"
#include "mc_cpu_reduce.h"
#include "utils/ucc_malloc.h"
#include <sys/types.h>

static ucc_status_t ucc_mc_cpu_mem_alloc_with_init(ucc_mc_buffer_header_t **ptr, size_t size);

static ucc_config_field_t ucc_mc_cpu_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cpu_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"CPU_ELEM_SIZE", "1024", "The size of each element in mc cpu mpool", ucc_offsetof(ucc_mc_cpu_config_t, cpu_elem_size),
    		UCC_CONFIG_TYPE_MEMUNITS},

    {"CPU_MAX_ELEMS", "8", "The max amount of elements in mc cpu mpool", ucc_offsetof(ucc_mc_cpu_config_t, cpu_max_elems),
    		UCC_CONFIG_TYPE_UINT},

    {NULL}
};

static ucc_status_t ucc_mc_cpu_init()
{
    ucc_strncpy_safe(ucc_mc_cpu.super.config->log_component.name,
                     ucc_mc_cpu.super.super.name,
                     sizeof(ucc_mc_cpu.super.config->log_component.name));
    ucc_spinlock_init(&ucc_mc_cpu.mpool_init_spinlock, 0);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_finalize()
{
	if (ucc_mc_cpu.mpool_init_flag){
		ucc_mpool_cleanup(&ucc_mc_cpu.mpool, 1);
		ucc_mc_cpu.mpool_init_flag = 0;
		ucc_mc_cpu.super.ops.mem_alloc = ucc_mc_cpu_mem_alloc_with_init;
	}
    ucc_spinlock_destroy(&ucc_mc_cpu.mpool_init_spinlock);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_mem_alloc(ucc_mc_buffer_header_t **ptr, size_t size)
{
	ucc_mc_buffer_header_t *h = NULL;
	if (size <= MC_CPU_CONFIG->cpu_elem_size){
	    h = (ucc_mc_buffer_header_t *) ucc_mpool_get(&ucc_mc_cpu.mpool);
	}
	if (!h) {
		// Slow path
		size_t size_with_h = size + sizeof(ucc_mc_buffer_header_t);
		h = (ucc_mc_buffer_header_t *) ucc_malloc(size_with_h, "mc cpu");
		if (!h) {
			mc_error(&ucc_mc_cpu.super, "failed to allocate %zd bytes", size_with_h);
			return UCC_ERR_NO_MEMORY;
		}
		h->from_pool = 0;
		h->addr = (void *) ((ptrdiff_t) h + sizeof(ucc_mc_buffer_header_t));
	}
	*ptr = h;
    return UCC_OK;
}

static ucc_status_t ucc_mc_cpu_chunk_alloc(ucc_mpool_t *mp, size_t *size_p, void **chunk_p) {
	*chunk_p = ucc_malloc(*size_p, "mc cpu"); // TODO: should I use hugeTableAlloc instead?
	if (!*chunk_p) {
		mc_error(&ucc_mc_cpu.super, "failed to allocate %zd bytes", *size_p);
		return UCC_ERR_NO_MEMORY;
	}

	return UCC_OK;
}

static void ucc_mc_cpu_chunk_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
	ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *) obj;
	h->from_pool = 1;
	h->addr = (void *) ((ptrdiff_t) h + sizeof(ucc_mc_buffer_header_t));
}

static void ucc_mc_cpu_chunk_release(ucc_mpool_t *mp, void *chunk) {
	ucc_free(chunk);
}

static ucc_mpool_ops_t ucc_mc_ops = {
    .chunk_alloc   = ucc_mc_cpu_chunk_alloc,
    .chunk_release = ucc_mc_cpu_chunk_release,
    .obj_init      = ucc_mc_cpu_chunk_init,
    .obj_cleanup   = NULL
};

static ucc_status_t ucc_mc_cpu_mem_alloc_with_init(ucc_mc_buffer_header_t **ptr, size_t size)
{
	ucc_spin_lock(&ucc_mc_cpu.mpool_init_spinlock);
	if (! ucc_mc_cpu.mpool_init_flag) {
		// TODO: currently only with thread multiple, need to change?
		ucc_status_t status = ucc_mpool_init(&ucc_mc_cpu.mpool, 0, sizeof(ucc_mc_buffer_header_t) + MC_CPU_CONFIG->cpu_elem_size,
				0, UCC_CACHE_LINE_SIZE, 1, MC_CPU_CONFIG->cpu_max_elems,
				&ucc_mc_ops, UCC_THREAD_MULTIPLE, "mc cpu mpool buffers");
		if (status != UCC_OK) {
			return status;
		}
		ucc_mc_cpu.super.ops.mem_alloc = ucc_mc_cpu_mem_alloc;
		ucc_mc_cpu.mpool_init_flag = 1;
	}
	ucc_spin_unlock(&ucc_mc_cpu.mpool_init_spinlock);
	return ucc_mc_cpu_mem_alloc(ptr, size);
}

static ucc_status_t ucc_mc_cpu_reduce(const void *src1, const void *src2,
                                      void *dst, size_t count,
                                      ucc_datatype_t dt, ucc_reduction_op_t op)
{
    switch(dt) {
    case UCC_DT_INT8:
        DO_DT_REDUCE_INT(int8_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_INT16:
        DO_DT_REDUCE_INT(int16_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_INT32:
        DO_DT_REDUCE_INT(int32_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_INT64:
        DO_DT_REDUCE_INT(int64_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_UINT8:
        DO_DT_REDUCE_INT(uint8_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_UINT16:
        DO_DT_REDUCE_INT(uint16_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_UINT32:
        DO_DT_REDUCE_INT(uint32_t, op, src1, src2, dst, count);
        break;
    case UCC_DT_UINT64:
        DO_DT_REDUCE_INT(uint64_t, op, src1, src2, dst, count);
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

static ucc_status_t ucc_mc_cpu_reduce_multi(const void *src1, const void *src2,
                                            void *dst, size_t size,
                                            size_t count, size_t stride,
                                            ucc_datatype_t dt,
                                            ucc_reduction_op_t op)
{
    int i;
    ucc_status_t st;

    //TODO implement efficient reduce_multi
    st = ucc_mc_cpu_reduce(src1, src2, dst, count, dt, op);
    for (i = 1; i < size; i++) {
        if (st != UCC_OK) {
            return st;
        }
        st = ucc_mc_cpu_reduce((void *)((ptrdiff_t)src2 + stride * i), dst, dst,
                               count, dt, op);
    }
    return st;
}

static ucc_status_t ucc_mc_cpu_mem_free(ucc_mc_buffer_header_t *ptr)
{
    if (!ptr->from_pool) {
    	ucc_free(ptr);
    }
    else {
    	ucc_mpool_put(ptr);
    }
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

static ucc_status_t ucc_mc_cpu_mem_query(const void *ptr, size_t length,
                                        ucc_mem_attr_t *mem_attr)
{
    if (ptr == NULL || length == 0) {
        mem_attr->mem_type = UCC_MEMORY_TYPE_HOST;
    }

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
    .super.init             = ucc_mc_cpu_init,
    .super.finalize         = ucc_mc_cpu_finalize,
    .super.ops.mem_query    = ucc_mc_cpu_mem_query,
    .super.ops.mem_alloc    = ucc_mc_cpu_mem_alloc_with_init,
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
    .mpool_init_flag = 0,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cpu.super.config_table,
                                &ucc_config_global_list);
