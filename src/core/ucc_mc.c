/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "components/mc/base/ucc_mc_base.h"
#include "ucc_mc.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

static const ucc_mc_ops_t *mc_ops[UCC_MEMORY_TYPE_LAST];

#define UCC_CHECK_MC_AVAILABLE(mc)                                             \
    do {                                                                       \
        if (NULL == mc_ops[mc]) {                                              \
            ucc_error("no components supported memory type %s available",      \
                    ucc_memory_type_names[mc]);                                \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

ucc_status_t ucc_mc_init()
{
    int            i, n_mcs;
    ucc_mc_base_t *mc;
    ucc_status_t   status;

    memset(mc_ops, 0, UCC_MEMORY_TYPE_LAST * sizeof(ucc_mc_ops_t *));
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
            status = mc->init();
            if (UCC_OK != status) {
                ucc_info("mc_init failed for component: %s, skipping (%d)",
                         mc->super.name, status);
                ucc_config_parser_release_opts(mc->config,
                                               mc->config_table.table);
                ucc_free(mc->config);
                continue;
            }
            ucc_debug("%s initialized", mc->super.name);
        }
        mc->ref_cnt++;
        mc_ops[mc->type] = &mc->ops;
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

ucc_status_t ucc_mc_type(const void *ptr, ucc_memory_type_t *mem_type)
{
    ucc_memory_type_t mt;
    ucc_status_t      status;
    ucc_mem_attr_t    mem_attr;

    /* TODO: consider using memory type cache from UCS */
    /* by default assume memory type host */
    *mem_type = UCC_MEMORY_TYPE_HOST;
    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    for (mt = UCC_MEMORY_TYPE_HOST + 1; mt < UCC_MEMORY_TYPE_LAST; mt++) {
        if (NULL != mc_ops[mt]) {
            status = mc_ops[mt]->mem_query(ptr, 0, &mem_attr);
            if (UCC_OK == status) {
                *mem_type = mem_attr.mem_type;
                /* found memory type for ptr */
                return UCC_OK;
            }
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_mc_query(const void *ptr, size_t length, ucc_mem_attr_t *mem_attr)
{
    ucc_status_t      status;
    ucc_memory_type_t mt;

    for (mt = UCC_MEMORY_TYPE_HOST + 1; mt < UCC_MEMORY_TYPE_LAST; mt++) {
        if (NULL != mc_ops[mt]) {
            status = mc_ops[mt]->mem_query(ptr, length, mem_attr);
            if (UCC_OK == status) {
                /* found memory type for ptr */
                return UCC_OK;
            }
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_mc_alloc(void **ptr, size_t size, ucc_memory_type_t mem_type)
{
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->mem_alloc(ptr, size);
}

ucc_status_t ucc_mc_reduce(const void *src1, const void *src2, void *dst,
                           size_t count, ucc_datatype_t dt,
                           ucc_reduction_op_t op, ucc_memory_type_t mem_type)
{
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->reduce(src1, src2, dst, count, dt, op);
}

ucc_status_t ucc_mc_free(void *ptr, ucc_memory_type_t mem_type)
{
    UCC_CHECK_MC_AVAILABLE(mem_type);
    return mc_ops[mem_type]->mem_free(ptr);
}

ucc_status_t ucc_mc_memcpy(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem)
{
    if (src_mem == UCC_MEMORY_TYPE_HOST &&
         dst_mem == UCC_MEMORY_TYPE_HOST) {
        UCC_CHECK_MC_AVAILABLE(UCC_MEMORY_TYPE_HOST);
        return mc_ops[UCC_MEMORY_TYPE_HOST]->mem_copy(dst, src, len,
                                                      UCC_MEMORY_TYPE_HOST,
                                                      UCC_MEMORY_TYPE_HOST);
    } else {
        /* take any non host MC component */
        ucc_memory_type_t mt = (dst_mem == UCC_MEMORY_TYPE_HOST) ?
            src_mem : dst_mem;
        UCC_CHECK_MC_AVAILABLE(mt);
        return mc_ops[mt]->mem_copy(dst, src, len, dst_mem, src_mem);
    }
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

ucc_status_t ucc_mc_reduce_multi(void *src1, void *src2, void *dst,
                                 size_t count, size_t size, size_t stride,
                                 ucc_datatype_t dtype, ucc_reduction_op_t op,
                                 ucc_memory_type_t mem_type)
{
    int i;
    if (size == 0) {
        return UCC_OK;
    }
    //TODO implemente efficient reduce_multi in the mc components
    ucc_mc_reduce(src1, src2, dst, size, dtype, op, mem_type);
    for (i = 1; i < count; i++) {
        ucc_mc_reduce((void *)((ptrdiff_t)src2 + stride * i), dst, dst, size,
                      dtype, op, mem_type);
    }
    return UCC_OK;
}
