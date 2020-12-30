/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_tl.h"
#include "utils/ucc_log.h"
#include <ucs/datastruct/khash.h>

KHASH_MAP_INIT_INT64(tl_obj, void*)
static khash_t(tl_obj) *tl_lib_map = NULL;
static khash_t(tl_obj) *tl_ctx_map = NULL;

ucc_config_field_t ucc_tl_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_base_config_table)},

};

ucc_config_field_t ucc_tl_context_config_table[] = {
    {NULL}
};

UCC_CLASS_INIT_FUNC(ucc_tl_lib_t, ucc_tl_iface_t *tl_iface,
                    const ucc_tl_lib_config_t *tl_config)
{
    self->iface         = tl_iface;
    self->super.log_component = tl_config->super.log_component;
    ucc_strncpy_safe(self->super.log_component.name,
                     tl_iface->tl_lib_config.name,
                     sizeof(self->super.log_component.name));
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_lib_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_lib_t, void);

UCC_CLASS_INIT_FUNC(ucc_tl_context_t, ucc_tl_lib_t *tl_lib)
{
    self->super.lib = &tl_lib->super;
    self->ref_count = 0;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_context_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_context_t, void);

static ucc_status_t ucc_tl_context_config_read(ucc_tl_lib_t *tl_lib,
                                               const char *full_prefix,
                                               ucc_tl_context_config_t **tl_config)
{
    ucc_status_t status;
    status = ucc_base_config_read(full_prefix,
                                  &tl_lib->iface->tl_context_config,
                                  (ucc_base_config_t **)tl_config);
    if (UCC_OK == status) {
        (*tl_config)->tl_lib = tl_lib;
    }
    return status;
}

static ucc_status_t ucc_tl_lib_config_read(ucc_tl_iface_t *iface,
                                           const char *full_prefix,
                                           ucc_tl_lib_config_t **tl_config)
{
    return ucc_base_config_read(full_prefix, &iface->tl_lib_config,
                                (ucc_base_config_t **)tl_config);
}

static ucc_tl_iface_t* get_tl_iface(ucc_tl_type_t type)
{
    int i;
    ucc_tl_iface_t *tl_iface;
    for (i=0; i < ucc_global_config.tl_framework.n_components; i++) {
        tl_iface = ucc_derived_of(ucc_global_config.tl_framework.components[i],
                                  ucc_tl_iface_t);
        if (tl_iface->type == type) {
            return tl_iface;
        }
    }
    return NULL;
}

ucc_status_t ucc_tl_context_get(ucc_context_t *ctx, ucc_tl_type_t type,
                                ucc_tl_context_t **tl_context)
{
    ucc_lib_info_t *lib = ctx->lib;
    khiter_t k;
    ucc_tl_lib_t **tl_libs;
    ucc_tl_lib_t *tl_lib;
    ucc_tl_context_t **tl_ctxs;
    ucc_tl_context_t *tl_ctx;
    ucc_base_lib_t *      b_lib;
    ucc_base_context_t *      b_ctx;
    int ret;
    ucc_status_t status;
    ucc_tl_iface_t *tl_iface;
    ucc_base_lib_params_t b_lib_params;
    ucc_base_context_params_t b_ctx_params;
    ucc_tl_lib_config_t  *tl_lib_config;
    ucc_tl_context_config_t  *tl_ctx_config;

    tl_iface = get_tl_iface(type);
    if (NULL == tl_iface) {
        /* Required tl dynamic component not available */
        return UCC_ERR_NOT_FOUND;
    }

    if (NULL == tl_lib_map) {
        /* Init hash to store the array of tl lib objects.
           The key of the hash is the ucc_lib_info_t pointer */
        tl_lib_map = kh_init(tl_obj);
    }

    k = kh_get(tl_obj, tl_lib_map, (int64_t)lib);
    if (k == kh_end(tl_lib_map)) {
        /* Given lib ptr is met for the first time:
           allocate the array of tl lib pointers and add to the hash */
        k = kh_put(tl_obj, tl_lib_map, (int64_t)lib, &ret);
        ucc_assert(k != kh_end(tl_lib_map));
        tl_libs = ucc_calloc(UCC_N_TLS, sizeof(ucc_tl_lib_t*), "tl_lib_ptr_array");
        if (!tl_libs) {
            return UCC_ERR_NO_MEMORY;
        }
        kh_value(tl_lib_map, k) = tl_libs;
    } else {
        /* Found lib ptr in the hash: get tl lib pointers array */
        tl_libs = kh_value(tl_lib_map, k);
    }

    if (NULL == tl_libs[ucc_ilog2(type)]) {
        /* Required TL was not initialized yet for this lib object */
        status = ucc_tl_lib_config_read(tl_iface, lib->full_prefix,
                                        &tl_lib_config);
        if (UCC_OK != status) {
            ucc_warn("failed to read TL \"%s\" lib configuration",
                     tl_iface->super.name);
            /* set to -1 to skip in the future if this TL is required by other CLs */
            tl_libs[ucc_ilog2(type)] = (void*)-1;
            return status;
        }
        ucc_copy_lib_params(&b_lib_params.params, &lib->params);
        status = tl_iface->lib.init(&b_lib_params, &tl_lib_config->super, &b_lib);
        ucc_base_config_release(&tl_lib_config->super);
        if (UCC_OK != status) {
            ucc_info("lib_init failed for tl component: %s",
                     tl_iface->super.name);
            tl_libs[ucc_ilog2(type)] = (void*)-1;
            return status;
        }
        tl_lib = ucc_derived_of(b_lib, ucc_tl_lib_t);
        tl_lib->ref_count = 0;
        tl_lib->lib = lib;
        tl_libs[ucc_ilog2(type)] = tl_lib;
    } else if ((void*)-1 == tl_libs[ucc_ilog2(type)]) {
        /* We already failed allocating this TL before */
        return UCC_ERR_NOT_FOUND;
    } else {
        /* TL lib is available in the hash */
        tl_lib = tl_libs[ucc_ilog2(type)];
    }

    if (NULL == tl_ctx_map) {
        /* Init hash to store the array of tl context objects.
           The key of the hash is the ucc_context_t pointer */
        tl_ctx_map = kh_init(tl_obj);
    }

    k = kh_get(tl_obj, tl_ctx_map, (int64_t)ctx);
    if (k == kh_end(tl_ctx_map)) {
        /* Given context ptr is met for the first time:
           allocate the array of tl context pointers and add to the hash */
        k = kh_put(tl_obj, tl_ctx_map, (int64_t)ctx, &ret);
        ucc_assert(k != kh_end(tl_ctx_map));
        tl_ctxs = ucc_calloc(UCC_N_TLS, sizeof(ucc_tl_context_t*), "tl_context_ptr_array");
        if (!tl_ctxs) {
            return UCC_ERR_NO_MEMORY;
        }
        kh_value(tl_ctx_map, k) = tl_ctxs;
    } else {
        /* Found context ptr in the hash: get tl context pointers array */
        tl_ctxs = kh_value(tl_ctx_map, k);
    }

    if (NULL == tl_ctxs[ucc_ilog2(type)]) {
        /* Required TL was not initialized yet for this context object */
        status = ucc_tl_context_config_read(tl_lib, lib->full_prefix,
                                            &tl_ctx_config);
        if (UCC_OK != status) {
            ucc_warn("failed to read TL \"%s\" lib configuration",
                     tl_iface->super.name);
            tl_libs[ucc_ilog2(type)] = (void*)-1;
            return status;
        }
        ucc_copy_context_params(&b_ctx_params.params, &ctx->params);
        b_ctx_params.prefix      = lib->full_prefix;
        b_ctx_params.thread_mode = lib->attr.thread_mode;

        status = tl_iface->context.create(&b_ctx_params, &tl_ctx_config->super, &b_ctx);
        ucc_base_config_release(&tl_ctx_config->super);
        if (UCC_OK != status) {
            ucc_info("context_init failed for tl component: %s",
                     tl_iface->super.name);
            tl_ctxs[ucc_ilog2(type)] = (void*)-1;
            return status;
        }
        tl_ctx = ucc_derived_of(b_ctx, ucc_tl_context_t);
        tl_ctx->ctx = ctx;
        tl_lib->ref_count++;
        tl_ctxs[ucc_ilog2(type)] = tl_ctx;
    } else if ((void*)-1 == tl_ctxs[ucc_ilog2(type)]) {
        return UCC_ERR_NOT_FOUND;
    } else {
        tl_ctx = tl_ctxs[ucc_ilog2(type)];
    }
    tl_ctx->ref_count++;
    *tl_context = tl_ctx;
    return UCC_OK;
}

ucc_status_t ucc_tl_context_put(ucc_tl_context_t *tl_context)
{
    ucc_tl_lib_t *tl_lib = ucc_derived_of(tl_context->super.lib,
                                          ucc_tl_lib_t);
    ucc_tl_iface_t *tl_iface = tl_lib->iface;
    tl_context->ref_count--;
    if (tl_context->ref_count == 0) {
        tl_iface->context.destroy(&tl_context->super);
        
        tl_lib->ref_count--;
    }
    return UCC_OK;
}
