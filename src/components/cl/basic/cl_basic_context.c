/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_context_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_context_config_t);
    ucc_config_names_array_t *tls = &cl_config->cl_lib->tls.array;
    ucc_status_t status;
    int          i;

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_context_t, cl_config,
                              params->context);
    if (tls->count == 1 && !strcmp(tls->names[0], "all")) {
        tls = &params->context->all_tls;
    }

    self->super.tl_ctxs = ucc_malloc(sizeof(ucc_tl_context_t*) * tls->count,
                               "cl_basic_tl_ctxs");
    if (!self->super.tl_ctxs) {
        cl_error(cl_config->cl_lib, "failed to allocate %zd bytes for tl_ctxs",
                 sizeof(ucc_tl_context_t**) * tls->count);
        return UCC_ERR_NO_MEMORY;
    }
    self->super.n_tl_ctxs = 0;
    for (i = 0; i < tls->count; i++) {
        status = ucc_tl_context_get(params->context, tls->names[i],
                                   &self->super.tl_ctxs[self->super.n_tl_ctxs]);
        if (UCC_OK != status) {
            cl_debug(cl_config->cl_lib,
                     "TL %s context is not available, skipping", tls->names[i]);
        } else {
            self->super.n_tl_ctxs++;
        }
    }
    if (0 == self->super.n_tl_ctxs) {
        cl_error(cl_config->cl_lib, "no TL contexts are available");
        ucc_free(self->super.tl_ctxs);
        self->super.tl_ctxs = NULL;
        return UCC_ERR_NOT_FOUND;
    }
    cl_debug(cl_config->cl_lib, "initialized cl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_context_t)
{
    int i;
    cl_debug(self->super.super.lib, "finalizing cl context: %p", self);
    for (i = 0; i < self->super.n_tl_ctxs; i++) {
        ucc_tl_context_put(self->super.tl_ctxs[i]);
    }
    ucc_free(self->super.tl_ctxs);
}

ucc_status_t ucc_cl_basic_mem_map(const ucc_base_context_t *context, int type, /* NOLINT */
                                  void *memh, void *tl_h) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_cl_basic_mem_unmap(const ucc_base_context_t *context, int type, /* NOLINT */
                                    void *tl_h) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_cl_basic_memh_pack(const ucc_base_context_t *context, /* NOLINT */
                                    int type, void *memh, void **packed_buffer) /* NOLINT */
{
    return UCC_ERR_NOT_SUPPORTED;
}

UCC_CLASS_DEFINE(ucc_cl_basic_context_t, ucc_cl_context_t);

ucc_status_t
ucc_cl_basic_get_context_attr(const ucc_base_context_t *context,
                              ucc_base_ctx_attr_t      *attr)
{
    const ucc_cl_basic_context_t *ctx =
        ucc_derived_of(context, ucc_cl_basic_context_t);
    ucc_base_ctx_attr_t tl_attr;
    ucc_status_t        status;
    int                 i;

    ucc_base_ctx_attr_clear(attr);

    /* CL BASIC reports topo_required if any of the TL available
     * TL contexts needs it
     */
    for (i = 0; i < ctx->super.n_tl_ctxs; i++) {
        memset(&tl_attr, 0, sizeof(tl_attr));
        status = UCC_TL_CTX_IFACE(ctx->super.tl_ctxs[i])
                    ->context.get_attr(&ctx->super.tl_ctxs[i]->super, &tl_attr);
        if (UCC_OK != status) {
            cl_error(ctx->super.super.lib, "failed to get %s ctx attr",
                     UCC_TL_CTX_IFACE(ctx->super.tl_ctxs[i])->super.name);
            return status;
        }
        if (tl_attr.topo_required) {
            attr->topo_required = 1;
            break;
        }
    }

    return UCC_OK;
}
