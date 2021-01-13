/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_tl_ucp_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_ucp_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_ucp_context_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super);
    tl_info(tl_context->lib, "initialized tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

ucc_status_t ucc_tl_ucp_team_create_test(ucc_base_team_t *tl_team)
{
    /* ucc_tl_ucp_team_t *team = ucc_derived_of(tl_team, ucc_tl_ucp_team_t); */
    return UCC_OK;
}

UCC_CLASS_DEFINE(ucc_tl_ucp_team_t, ucc_tl_team_t);
