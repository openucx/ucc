/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_team_t, ucc_base_context_t *cl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_cl_basic_context_t *ctx =
        ucc_derived_of(cl_context, ucc_cl_basic_context_t);
    ucc_status_t     status;
    ucc_base_team_t *b_team;
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super);
    status = UCC_TL_CTX_IFACE(ctx->tl_ucp_ctx)
                 ->team.create_post(&ctx->tl_ucp_ctx->super, params, &b_team);
    if (UCC_OK != status) {
        self->tl_ucp_team = NULL;
        cl_error(cl_context->lib, "tl ucp team create post failed");
        return status;
    }
    self->tl_ucp_team = ucc_derived_of(b_team, ucc_tl_team_t);
    cl_info(cl_context->lib, "posted cl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_team_t)
{
    cl_info(self->super.super.context->lib, "finalizing cl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_basic_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_cl_basic_team_t, ucc_cl_team_t);

ucc_status_t ucc_cl_basic_team_destroy(ucc_base_team_t *cl_team)
{
    ucc_cl_basic_team_t    *team = ucc_derived_of(cl_team, ucc_cl_basic_team_t);
    ucc_cl_basic_context_t *ctx  = UCC_CL_BASIC_TEAM_CTX(team);
    ucc_status_t            status;
    if (team->tl_ucp_team) {
        if (UCC_OK !=
            (status = UCC_TL_CTX_IFACE(ctx->tl_ucp_ctx)
                          ->team.destroy(&team->tl_ucp_team->super))) {
            return status;
        }
        team->tl_ucp_team = NULL;
    }
    UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_basic_team_t)(cl_team);
    return UCC_OK;
}

ucc_status_t ucc_cl_basic_team_create_test(ucc_base_team_t *cl_team)
{
    ucc_cl_basic_team_t    *team = ucc_derived_of(cl_team, ucc_cl_basic_team_t);
    ucc_cl_basic_context_t *ctx  = UCC_CL_BASIC_TEAM_CTX(team);
    ucc_status_t            status;
    status = UCC_TL_CTX_IFACE(ctx->tl_ucp_ctx)
                 ->team.create_test(&team->tl_ucp_team->super);
    if (UCC_OK == status) {
        cl_info(ctx->super.super.lib, "initialized cl team: %p", team);
    } else if (UCC_INPROGRESS != status) {
        cl_error(ctx->super.super.lib, "failed to create tl ucp team");
    }
    return status;
}
