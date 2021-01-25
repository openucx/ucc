/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_status_t status;
    const ucc_cl_context_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_context_config_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_context_t, cl_config->cl_lib,
                              params->context);
    status = ucc_tl_context_get(params->context, UCC_TL_UCP,
                                &self->tl_ucp_ctx);
    if (UCC_OK != status) {
        cl_warn(cl_config->cl_lib, "TL UCP context is not available, CL BASIC can't proceed");
        return UCC_ERR_NOT_FOUND;
    }
    cl_info(cl_config->cl_lib, "initialized cl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_context_t)
{
    cl_info(self->super.super.lib, "finalizing cl context: %p", self);
    ucc_tl_context_put(self->tl_ucp_ctx);
}

UCC_CLASS_DEFINE(ucc_cl_basic_context_t, ucc_cl_context_t);

ucc_status_t ucc_cl_basic_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                                           ucc_base_attr_t *attr)             /* NOLINT */
{
    /* TODO */
    return UCC_ERR_NOT_IMPLEMENTED;
}
