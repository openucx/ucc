/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "utils/ucc_malloc.h"
#include "utils/arch/cpu.h"
#include "cl_hier_coll.h"

UCC_CLASS_INIT_FUNC(ucc_cl_hier_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t         *config)
{
    const ucc_cl_context_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_context_config_t);
    ucc_cl_hier_lib_t        *lib = ucc_derived_of(cl_config->cl_lib,
                                                   ucc_cl_hier_lib_t);
    ucc_config_names_array_t *tls = &lib->tls.array;
    ucc_status_t              status;
    int                       i;

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_context_t, cl_config,
                              params->context);
    if (tls->count == 1 && !strcmp(tls->names[0], "all")) {
        tls = &params->context->all_tls;
    }

    self->super.tl_ctxs =
        ucc_malloc(sizeof(ucc_tl_context_t *) * tls->count, "cl_hier_tl_ctxs");
    if (!self->super.tl_ctxs) {
        cl_error(cl_config->cl_lib, "failed to allocate %zd bytes for tl_ctxs",
                 sizeof(ucc_tl_context_t **) * tls->count);
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
        status = UCC_ERR_NOT_FOUND;
        goto out;
    }

    status = ucc_mpool_init(&self->sched_mp, 0, sizeof(ucc_cl_hier_schedule_t),
                            0, UCC_CACHE_LINE_SIZE, 2, UINT_MAX,
                            &ucc_coll_task_mpool_ops, params->thread_mode,
                            "cl_hier_sched_mp");
    if (UCC_OK != status) {
        cl_error(cl_config->cl_lib, "failed to initialize cl_hier_sched mpool");
        goto out;
    }

    cl_debug(cl_config->cl_lib, "initialized cl context: %p", self);
    return UCC_OK;

out:
    ucc_free(self->super.tl_ctxs);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_hier_context_t)
{
    int i;
    cl_debug(self->super.super.lib, "finalizing cl context: %p", self);

    ucc_mpool_cleanup(&self->sched_mp, 1);
    for (i = 0; i < self->super.n_tl_ctxs; i++) {
        ucc_tl_context_put(self->super.tl_ctxs[i]);
    }
    ucc_free(self->super.tl_ctxs);
}

UCC_CLASS_DEFINE(ucc_cl_hier_context_t, ucc_cl_context_t);

ucc_status_t
ucc_cl_hier_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
    attr->topo_required = 1;
    return UCC_OK;
}

/* NOLINTBEGIN */
ucc_status_t ucc_cl_hier_mem_map(const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
                                 ucc_mem_map_memh_t *memh, ucc_mem_map_tl_t *tl_h)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_cl_hier_mem_unmap(const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
                                   ucc_mem_map_tl_t *tl_h)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_cl_hier_memh_pack(const ucc_base_context_t *context,
                                   ucc_mem_map_mode_t mode, ucc_mem_map_tl_t *tl_h,
                                   void **packed_buffer)
{
    return UCC_ERR_NOT_SUPPORTED;
}
/* NOLINTEND */

