/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "components/mc/base/ucc_mc_base.h"
#include "ucc_mc.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

#ifdef HAVE_PROFILING_MC
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif
#define UCC_MC_PROFILE_FUNC UCC_PROFILE_FUNC

static const ucc_mc_ops_t *mc_ops[UCC_MEMORY_TYPE_LAST];
static const ucc_ee_ops_t *ee_ops[UCC_EE_LAST];

#define UCC_CHECK_MC_AVAILABLE(mc)                                             \
    do {                                                                       \
        if (ucc_unlikely(NULL == mc_ops[mc])) {                                \
            ucc_error("no components supported memory type %s available",      \
                      ucc_memory_type_names[mc]);                              \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

#define UCC_CHECK_EE_AVAILABLE(ee)                                             \
    do {                                                                       \
        if (NULL == ee_ops[ee]) {                                              \
            ucc_error("no components supported ee %s available",               \
                      ucc_ee_type_names[ee]);                                  \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

ucc_status_t ucc_mc_init(const ucc_mc_params_t *mc_params)
{
    int            i, n_mcs;
    ucc_mc_base_t *mc;
    ucc_status_t   status;
    ucc_mc_attr_t attr;

    memset(mc_ops, 0, UCC_MEMORY_TYPE_LAST * sizeof(ucc_mc_ops_t *));
    memset(ee_ops, 0, UCC_EE_LAST * sizeof(ucc_ee_ops_t *));
    n_mcs = ucc_global_config.mc_framework.n_components;
    for (i = 0; i < n_mcs; i++) {
        mc = ucc_derived_of(ucc_global_config.mc_framework.components[i],
                            ucc_mc_base_t);
        if (mc->ref_cnt == 0) {
            mc->config = ucc_malloc(mc->config_table.size);
            if (!mc->config) {
                ucc_error("failed to allocate %zd bytes for mc config",
                          mc->config_table.size);
                continue;
            }
            status = ucc_config_parser_fill_opts(
                mc->config, mc->config_table.table, "UCC_",
                mc->config_table.prefix, 1);
            if (UCC_OK != status) {
                ucc_info("failed to parse config for component: %s (%d)",
                         mc->super.name, status);
                ucc_free(mc->config);
                continue;
            }
            status = mc->init(mc_params);
            if (UCC_OK != status) {
                ucc_info("mc_init failed for component: %s, skipping (%d)",
                         mc->super.name, status);
                ucc_config_parser_release_opts(mc->config,
                                               mc->config_table.table);
                ucc_free(mc->config);
                continue;
            }
            ucc_debug("%s initialized", mc->super.name);
        } else {
            attr.field_mask = UCC_MC_ATTR_FIELD_THREAD_MODE;
            status = mc->get_attr(&attr);
            if (status != UCC_OK) {
                return status;
            }
            if (attr.thread_mode < mc_params->thread_mode) {
                ucc_warn("memory component %s was allready initilized with "
                         "different thread mode: current tm %d, provided tm %d",
                         mc->super.name, attr.thread_mode,
                         mc_params->thread_mode);
            }
        }
        mc->ref_cnt++;
        mc_ops[mc->type] = &mc->ops;
        ee_ops[mc->ee_type] = &mc->ee_ops;
    }

    return UCC_OK;
}

ucc_status_t ucc_mc_available(ucc_memory_type_t mem_type)
{
    if (NULL == mc_ops[mem_type]) {
        return UCC_ERR_NOT_FOUND;
    }

    return UCC_OK;
}

ucc_status_t ucc_mc_get_mem_attr(const void *ptr, ucc_mem_attr_t *mem_attr)
{
    ucc_status_t      status;
    ucc_memory_type_t mt;

    mem_attr->mem_type     = UCC_MEMORY_TYPE_HOST;
    mem_attr->base_address = (void *)ptr;
    if (ptr == NULL) {
        mem_attr->alloc_length = 0;
        return UCC_OK;
    }

    mt = (ucc_memory_type_t)(UCC_MEMORY_TYPE_HOST + 1);
    for (; mt < UCC_MEMORY_TYPE_LAST; mt++) {
        if (NULL != mc_ops[mt]) {
            status = mc_ops[mt]->mem_query(ptr, mem_attr);
            if (UCC_OK == status) {
                return UCC_OK;
            }
        }
    }
    return UCC_OK;
}

UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_alloc, (h_ptr, size, mem_type),
                    ucc_mc_buffer_header_t **h_ptr, size_t size,
                    ucc_memory_type_t mem_type)
{
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->mem_alloc(h_ptr, size);
}

UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_reduce,
                 (src1, src2, dst, count, dt, op, mem_type),
                 const void *src1, const void *src2, void *dst,
                 size_t count, ucc_datatype_t dt,
                 ucc_reduction_op_t op, ucc_memory_type_t mem_type)

{
    if (count == 0) {
        return UCC_OK;
    }
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->reduce(src1, src2, dst, count, dt, op);
}

UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_reduce_multi,
                    (src1, src2, dst, n_vectors, count, stride, dtype, op, mem_type),
                    void *src1, void *src2, void *dst, size_t n_vectors,
                    size_t count, size_t stride, ucc_datatype_t dtype,
                    ucc_reduction_op_t op, ucc_memory_type_t mem_type)
{
    if (count == 0) {
        return UCC_OK;
    }
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->reduce_multi(src1, src2, dst, n_vectors, count, stride,
                                          dtype, op);
}

UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_reduce_multi_alpha,
                    (src1, src2, dst, n_vectors, count, stride, dtype, reduce_op,
                     vector_op, alpha, mem_type),
                    void *src1, void *src2, void *dst, size_t n_vectors,
                    size_t count, size_t stride, ucc_datatype_t dtype,
                    ucc_reduction_op_t reduce_op, ucc_reduction_op_t vector_op,
                    double alpha, ucc_memory_type_t mem_type)
{
    if (count == 0) {
        return UCC_OK;
    }
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->reduce_multi_alpha(src1, src2, dst, n_vectors,
                                                count, stride, dtype, reduce_op,
                                                vector_op, alpha);
}

ucc_status_t ucc_mc_free(ucc_mc_buffer_header_t *h_ptr)
{
    UCC_CHECK_MC_AVAILABLE(h_ptr->mt);
    return mc_ops[h_ptr->mt]->mem_free(h_ptr);
}

UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_memcpy,
                    (dst, src, len, dst_mem, src_mem), void *dst,
                    const void *src, size_t len, ucc_memory_type_t dst_mem,
                    ucc_memory_type_t src_mem)

{
    ucc_memory_type_t mt;
    if (src_mem == UCC_MEMORY_TYPE_UNKNOWN ||
        dst_mem == UCC_MEMORY_TYPE_UNKNOWN) {
        return UCC_ERR_INVALID_PARAM;
    } else if (src_mem == UCC_MEMORY_TYPE_HOST &&
               dst_mem == UCC_MEMORY_TYPE_HOST) {
        UCC_CHECK_MC_AVAILABLE(UCC_MEMORY_TYPE_HOST);
        return mc_ops[UCC_MEMORY_TYPE_HOST]->memcpy(dst, src, len,
                                                    UCC_MEMORY_TYPE_HOST,
                                                    UCC_MEMORY_TYPE_HOST);
    }
    /* take any non host MC component */
    mt = (dst_mem == UCC_MEMORY_TYPE_HOST) ? src_mem : dst_mem;
    UCC_CHECK_MC_AVAILABLE(mt);
    return mc_ops[mt]->memcpy(dst, src, len, dst_mem, src_mem);
}

ucc_status_t ucc_mc_flush(ucc_memory_type_t mem_type)
{
    UCC_CHECK_MC_AVAILABLE(mem_type);
    if (mc_ops[mem_type]->flush) {
        return mc_ops[mem_type]->flush();
    }
    return UCC_OK;
}

ucc_status_t ucc_mc_finalize()
{
   ucc_memory_type_t  mt;
   ucc_mc_base_t     *mc;

    for (mt = UCC_MEMORY_TYPE_HOST; mt < UCC_MEMORY_TYPE_LAST; mt++) {
        if (NULL != mc_ops[mt]) {
            mc = ucc_container_of(mc_ops[mt], ucc_mc_base_t, ops);
            mc->ref_cnt--;
            if (mc->ref_cnt == 0) {
                mc->finalize();
                ucc_config_parser_release_opts(mc->config,
                                               mc->config_table.table);
                ucc_free(mc->config);
                mc_ops[mt] = NULL;
            }
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_mc_ee_task_post(void *ee_context, ucc_ee_type_t ee_type,
                                 void **ee_task)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_task_post(ee_context, ee_task);
}

ucc_status_t ucc_mc_ee_task_query(void *ee_task, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_task_query(ee_task);
}

ucc_status_t ucc_mc_ee_task_end(void *ee_task, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_task_end(ee_task);
}

ucc_status_t ucc_mc_ee_create_event(void **event, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_create_event(event);
}

ucc_status_t ucc_mc_ee_destroy_event(void *event, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_destroy_event(event);
}

ucc_status_t ucc_mc_ee_event_post(void *ee_context, void *event,
                                  ucc_ee_type_t ee_type)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_event_post(ee_context, event);
}

ucc_status_t ucc_mc_ee_event_test(void *event, ucc_ee_type_t ee_type)
{
    UCC_CHECK_EE_AVAILABLE(ee_type);
    return ee_ops[ee_type]->ee_event_test(event);
}
