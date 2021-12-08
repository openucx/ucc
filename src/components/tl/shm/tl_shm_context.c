/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "tl_shm_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include <limits.h>

UCC_CLASS_INIT_FUNC(ucc_tl_shm_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_status_t status;
	ucc_tl_shm_context_config_t *tl_shm_config =
	    ucc_derived_of(config, ucc_tl_shm_context_config_t);

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_shm_config->super.tl_lib,
	                              params->context);

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_shm_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX, NULL,
                            params->thread_mode, "tl_shm_req_mp");

    if (UCC_OK != status) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_shm_req mpool");
        return status;
    }
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
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
