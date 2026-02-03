/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "components/mc/base/ucc_mc_base.h"
#include "ucc_mc.h"
#include "ucc_mc_user_component.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"

#ifdef HAVE_PROFILING_MC
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif
#define UCC_MC_PROFILE_FUNC UCC_PROFILE_FUNC

static const ucc_mc_ops_t *mc_ops[UCC_MEMORY_TYPE_LAST];

static inline const ucc_mc_ops_t *ucc_mc_get_ops(ucc_memory_type_t mem_type)
{
    if (mem_type < UCC_MEMORY_TYPE_LAST) {
        /* Built-in component - direct array lookup */
        return mc_ops[mem_type];
    } else {
        /* User component - lookup in registry */
        ucc_mc_user_component_entry_t *uc = ucc_mc_user_component_get_entry(mem_type);
        return uc ? &uc->mc->ops : NULL;
    }
}

#define UCC_CHECK_MC_AVAILABLE(mc)                                             \
    do {                                                                       \
        const ucc_mc_ops_t *_ops = ucc_mc_get_ops(mc);                         \
        if (ucc_unlikely(NULL == _ops)) {                                      \
            if ((mc) < UCC_MEMORY_TYPE_LAST) {                                 \
                ucc_error(                                                     \
                    "memory type %s not initialized",                          \
                    ucc_memory_type_names[mc]);                                \
            } else {                                                           \
                ucc_error(                                                     \
                    "MC user component with memory type %d not registered",    \
                    mc);                                                       \
            }                                                                  \
            return UCC_ERR_NOT_SUPPORTED;                                      \
        }                                                                      \
    } while (0)

ucc_status_t ucc_mc_init(const ucc_mc_params_t *mc_params)
{
    int               i, n_mcs;
    ucc_mc_base_t    *mc;
    ucc_status_t      status;
    ucc_mc_attr_t     attr;
    int               is_user_component;
    ucc_memory_type_t assigned_type;

    memset(mc_ops, 0, UCC_MEMORY_TYPE_LAST * sizeof(ucc_mc_ops_t *));
    n_mcs = ucc_global_config.mc_framework.n_components;
    for (i = 0; i < n_mcs; i++) {
        mc = ucc_derived_of(ucc_global_config.mc_framework.components[i],
                            ucc_mc_base_t);

        /* Detect if this is a user component (loaded from user component path).
         * User components are marked with UCC_MEMORY_TYPE_LAST by the loader
         * before being assigned a unique dynamic type. */
        is_user_component = (mc->type >= UCC_MEMORY_TYPE_LAST);

        if (mc->ref_cnt == 0) {
            mc->config = ucc_malloc(mc->config_table.size);
            if (!mc->config) {
                ucc_error("failed to allocate %zd bytes for mc config",
                          mc->config_table.size);
                continue;
            }
            status = ucc_config_parser_fill_opts(
                mc->config, &mc->config_table, "UCC_", 1);
            if (UCC_OK != status) {
                ucc_debug("failed to parse config for mc: %s (%d)",
                          mc->super.name, status);
                ucc_free(mc->config);
                continue;
            }
            status = mc->init(mc_params);
            if (UCC_OK != status) {
                ucc_debug("mc_init failed for component: %s, skipping (%d)",
                          mc->super.name, status);
                ucc_config_parser_release_opts(mc->config,
                                               mc->config_table.table);
                ucc_free(mc->config);
                continue;
            }
            ucc_debug("mc %s initialized", mc->super.name);

            /* Register user components (only on first init to avoid duplicates) */
            if (is_user_component) {
                status = ucc_mc_user_component_register(mc, &assigned_type);
                if (status != UCC_OK) {
                    ucc_error(
                        "failed to register autodiscovered user component %s",
                        mc->super.name);
                    ucc_config_parser_release_opts(
                        mc->config, mc->config_table.table);
                    ucc_free(mc->config);
                    continue;
                }
                mc->type = assigned_type;
                ucc_info("MC user component '%s' registered with memory_type=%d",
                         mc->super.name, assigned_type);
            }
        } else {
            attr.field_mask = UCC_MC_ATTR_FIELD_THREAD_MODE;
            status = mc->get_attr(&attr);
            if (status != UCC_OK) {
                return status;
            }
            if (attr.thread_mode < mc_params->thread_mode) {
                ucc_info("mc %s was allready initilized with "
                         "different thread mode: current tm %d, provided tm %d",
                         mc->super.name, attr.thread_mode,
                         mc_params->thread_mode);
            }
        }
        mc->ref_cnt++;

        /* Register built-in components in ops table (safe to do every time,
         * needed because memset clears mc_ops on each ucc_mc_init call) */
        if (!is_user_component) {
            mc_ops[mc->type] = &mc->ops;
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_mc_available(ucc_memory_type_t mem_type)
{
    const ucc_mc_ops_t *ops;

    mem_type = (mem_type == UCC_MEMORY_TYPE_CUDA_MANAGED) ?
                   UCC_MEMORY_TYPE_CUDA : mem_type;

    ops = ucc_mc_get_ops(mem_type);
    if (NULL == ops) {
        return UCC_ERR_NOT_FOUND;
    }

    return UCC_OK;
}

/* Context for memory query iteration */
typedef struct {
    const void     *ptr;
    ucc_mem_attr_t *mem_attr;
    ucc_status_t    result;
} ucc_mc_mem_query_ctx_t;

static ucc_status_t ucc_mc_mem_query_cb(
    ucc_mc_user_component_entry_t *user_component, void *ctx)
{
    ucc_mc_mem_query_ctx_t *query_ctx = (ucc_mc_mem_query_ctx_t *)ctx;
    ucc_status_t            status;

    if (query_ctx->result == UCC_OK) {
        /* Already claimed by a previous component */
        return UCC_OK;
    }
    if (user_component->mc->ops.mem_query) {
        status = user_component->mc->ops.mem_query(
            query_ctx->ptr, query_ctx->mem_attr);
        if (UCC_OK == status) {
            query_ctx->result = UCC_OK;
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_mc_get_mem_attr(const void *ptr, ucc_mem_attr_t *mem_attr)
{
    ucc_status_t           status;
    ucc_memory_type_t      mt;
    const ucc_mc_ops_t    *ops;
    ucc_mc_mem_query_ctx_t query_ctx;

    mem_attr->mem_type     = UCC_MEMORY_TYPE_HOST;
    mem_attr->base_address = (void *)ptr;
    if (ptr == NULL) {
        mem_attr->alloc_length = 0;
        return UCC_OK;
    }

    /* Check builtin memory types */
    mt = (ucc_memory_type_t)(UCC_MEMORY_TYPE_HOST + 1);
    for (; mt < UCC_MEMORY_TYPE_LAST; mt++) {
        ops = ucc_mc_get_ops(mt);
        if (NULL != ops) {
            status = ops->mem_query(ptr, mem_attr);
            if (UCC_OK == status) {
                return UCC_OK;
            }
        }
    }

    /* Check user component memory types */
    query_ctx.ptr      = ptr;
    query_ctx.mem_attr = mem_attr;
    query_ctx.result   = UCC_ERR_NOT_FOUND;
    ucc_mc_user_component_iterate(ucc_mc_mem_query_cb, &query_ctx);
    if (query_ctx.result == UCC_OK) {
        return UCC_OK;
    }

    return UCC_OK;
}

ucc_status_t ucc_mc_get_attr(ucc_mc_attr_t *attr, ucc_memory_type_t mem_type)
{
    ucc_memory_type_t mt = (mem_type == UCC_MEMORY_TYPE_CUDA_MANAGED)
                               ? UCC_MEMORY_TYPE_CUDA
                               : mem_type;
    ucc_mc_base_t                 *mc;
    ucc_mc_user_component_entry_t *uc;

    UCC_CHECK_MC_AVAILABLE(mt);

    if (mt < UCC_MEMORY_TYPE_LAST) {
        mc = ucc_container_of(mc_ops[mt], ucc_mc_base_t, ops);
        return mc->get_attr(attr);
    } else {
        uc = ucc_mc_user_component_get_entry(mt);
        if (uc && uc->mc->get_attr) {
            return uc->mc->get_attr(attr);
        }
        return UCC_ERR_NOT_SUPPORTED;
    }
}

/* TODO: add the flexbility to bypass the mpool if the user asks for it */
UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_alloc, (h_ptr, size, mem_type),
                    ucc_mc_buffer_header_t **h_ptr, size_t size,
                    ucc_memory_type_t mem_type)
{
    ucc_memory_type_t   mt = (mem_type == UCC_MEMORY_TYPE_CUDA_MANAGED) ?
                                 UCC_MEMORY_TYPE_CUDA : mem_type;
    const ucc_mc_ops_t *ops;

    UCC_CHECK_MC_AVAILABLE(mt);
    ops = ucc_mc_get_ops(mt);
    return ops->mem_alloc(h_ptr, size, mem_type);
}

ucc_status_t ucc_mc_free(ucc_mc_buffer_header_t *h_ptr)
{
    ucc_memory_type_t   mt = (h_ptr->mt == UCC_MEMORY_TYPE_CUDA_MANAGED) ?
                                 UCC_MEMORY_TYPE_CUDA : h_ptr->mt;
    const ucc_mc_ops_t *ops;

    UCC_CHECK_MC_AVAILABLE(mt);
    ops = ucc_mc_get_ops(mt);
    return ops->mem_free(h_ptr);
}

UCC_MC_PROFILE_FUNC(ucc_status_t, ucc_mc_memcpy,
                    (dst, src, len, dst_mem, src_mem), void *dst,
                    const void *src, size_t len, ucc_memory_type_t dst_mem,
                    ucc_memory_type_t src_mem)

{
    ucc_memory_type_t   mt;
    const ucc_mc_ops_t *ops;

    if (src_mem == UCC_MEMORY_TYPE_UNKNOWN ||
        dst_mem == UCC_MEMORY_TYPE_UNKNOWN) {
        return UCC_ERR_INVALID_PARAM;
    } else if (src_mem == UCC_MEMORY_TYPE_HOST &&
               dst_mem == UCC_MEMORY_TYPE_HOST) {
        UCC_CHECK_MC_AVAILABLE(UCC_MEMORY_TYPE_HOST);
        ops = ucc_mc_get_ops(UCC_MEMORY_TYPE_HOST);
        return ops->memcpy(dst, src, len,
                          UCC_MEMORY_TYPE_HOST,
                          UCC_MEMORY_TYPE_HOST);
    }
    /* take any non host MC component */
    mt = (dst_mem == UCC_MEMORY_TYPE_HOST) ? src_mem : dst_mem;
    mt = (mt == UCC_MEMORY_TYPE_CUDA_MANAGED) ? UCC_MEMORY_TYPE_CUDA : mt;
    UCC_CHECK_MC_AVAILABLE(mt);
    ops = ucc_mc_get_ops(mt);
    return ops->memcpy(dst, src, len, dst_mem, src_mem);
}

ucc_status_t ucc_mc_memset(void *ptr, int value, size_t size,
                           ucc_memory_type_t mem_type)
{
    const ucc_mc_ops_t *ops;

    mem_type = (mem_type == UCC_MEMORY_TYPE_CUDA_MANAGED) ?
                   UCC_MEMORY_TYPE_CUDA : mem_type;

    UCC_CHECK_MC_AVAILABLE(mem_type);
    ops = ucc_mc_get_ops(mem_type);
    return ops->memset(ptr, value, size);
}

ucc_status_t ucc_mc_flush(ucc_memory_type_t mem_type)
{
    const ucc_mc_ops_t *ops;

    mem_type = (mem_type == UCC_MEMORY_TYPE_CUDA_MANAGED) ?
                   UCC_MEMORY_TYPE_CUDA : mem_type;

    UCC_CHECK_MC_AVAILABLE(mem_type);
    ops = ucc_mc_get_ops(mem_type);
    if (ops->flush) {
        return ops->flush();
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

    ucc_mc_user_component_finalize_all();

    return UCC_OK;
}
