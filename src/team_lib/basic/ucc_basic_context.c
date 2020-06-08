/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "ucc_basic_context.h"

ucc_status_t ucc_basic_context_create(ucc_team_lib_t *tl_lib,
                                      const ucc_context_params_t *params,
                                      const ucc_tl_context_config_t *config,
                                      ucc_tl_context_t **tl_context)
{
    ucc_tl_basic_context_t *ctx;
    ucc_status_t status;
    ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    *tl_context = &ctx->super;
    return UCC_OK;

error:
    return status;
}

void ucc_basic_context_destroy(ucc_tl_context_t *tl_context)
{
    ucc_tl_basic_context_t *ctx = ucs_derived_of(tl_context, ucc_tl_basic_context_t);
    free(ctx);
}
