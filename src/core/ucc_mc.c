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
                mc->config, mc->config_table.table, "UCC_", NULL, 1);
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

    /* TODO: consider using memory type cache from UCS */
    /* by default assume memory type host */
    *mem_type = UCC_MEMORY_TYPE_HOST;
    for (mt = UCC_MEMORY_TYPE_HOST + 1; mt < UCC_MEMORY_TYPE_LAST; mt++) {
        if (NULL != mc_ops[mt]) {
            status = mc_ops[mt]->mem_type(ptr, mem_type);
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
    ucc_status_t status;

    if (NULL == mc_ops[mem_type]) {
        ucc_error("no components supported memory type %s available",
                  ucc_memory_type_names[mem_type]);
        return UCC_ERR_NOT_SUPPORTED;
    }
    status = mc_ops[mem_type]->mem_alloc(ptr, size);

    return status;
}

ucc_status_t ucc_mc_reduce(const void *src1, ucc_memory_type_t src1_mt,
                           const void *src2, ucc_memory_type_t src2_mt,
                           void *dst, ucc_memory_type_t dst_mt, size_t count,
                           ucc_datatype_t dt, ucc_reduction_op_t op)
{
    //TODO

    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_mc_free(void *ptr, ucc_memory_type_t mem_type)
{
    ucc_status_t status;

    if (NULL == mc_ops[mem_type]) {
        ucc_error("no components supported memory type %s available",
                  ucc_memory_type_names[mem_type]);
        return UCC_ERR_NOT_SUPPORTED;
    }
    status = mc_ops[mem_type]->mem_free(ptr);

    return status;
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
