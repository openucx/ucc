/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"

UCC_CLASS_INIT_FUNC(ucc_tl_shm_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
	ucc_tl_shm_context_config_t *tl_shm_config =
	    ucc_derived_of(config, ucc_tl_shm_context_config_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_shm_config->super.tl_lib,
	                              params->context);

    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_context_t)
{
    tl_info(&self->super, "finalizing tl context: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_shm_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_shm_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }
    attr->topo_required = 1;
    return UCC_OK;
}
