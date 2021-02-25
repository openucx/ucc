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
    ucc_status_t status;
    ucc_base_team_params_t *b_params;
    int nteams = 0;

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super);
    b_params = ucc_malloc(sizeof(*b_params), "base team params");
    if (!b_params) {
        cl_error(cl_context->lib, "failed to allocate %zd bytes for params",
                 sizeof(*b_params));
        return UCC_ERR_NO_MEMORY;
    }
    memcpy(b_params, params, sizeof(ucc_base_team_params_t));
    b_params->scope    = UCC_CL_BASIC;
    b_params->scope_id = 0;
    self->tl_ucp_team  = NULL;
    self->tl_nccl_team = NULL;

    status = ucc_team_create_multiple_req_alloc(&self->team_create_req, 2);
    if (UCC_OK != status) {
        ucc_free(b_params);
        return status;
    }
    ucc_assert(ctx->tl_ucp_ctx != NULL);
    self->team_create_req->contexts[nteams] = ctx->tl_ucp_ctx;
    self->team_create_req->params[nteams] = b_params;
    nteams += 1;
    if (ctx->tl_nccl_ctx != NULL) {
        self->team_create_req->contexts[nteams] = ctx->tl_nccl_ctx;
        self->team_create_req->params[nteams] = b_params;
        nteams += 1;
    }
    self->team_create_req->n_teams = nteams;

    status = ucc_tl_team_create_multiple(self->team_create_req);
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
    if (team->tl_nccl_team) {
        if (UCC_OK !=
            (status = UCC_TL_CTX_IFACE(ctx->tl_nccl_ctx)
                          ->team.destroy(&team->tl_nccl_team->super))) {
            return status;
        }
        team->tl_nccl_team = NULL;
    }
    UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_basic_team_t)(cl_team);
    return UCC_OK;
}

ucc_status_t ucc_cl_basic_team_create_test(ucc_base_team_t *cl_team)
{
    ucc_cl_basic_team_t    *team = ucc_derived_of(cl_team, ucc_cl_basic_team_t);
    ucc_cl_basic_context_t *ctx  = UCC_CL_BASIC_TEAM_CTX(team);
    ucc_status_t            status;

    status = ucc_tl_team_create_multiple(team->team_create_req);
    if (status == UCC_OK) {
        if ((ctx->tl_nccl_ctx != NULL) &&
            (team->team_create_req->teams_status[1] == UCC_OK)) {
            team->tl_nccl_team = team->team_create_req->teams[1];
            cl_info(ctx->super.super.lib, "initialized nccl team");
        }
        if (team->team_create_req->teams_status[0] != UCC_OK)  {
            if (team->tl_nccl_team) {
                UCC_TL_CTX_IFACE(ctx->tl_nccl_ctx)
                                ->team.destroy(&team->tl_nccl_team->super);
                team->tl_nccl_team = NULL;
            }
            team->tl_ucp_team = NULL;
            cl_error(ctx->super.super.lib, "failed to create tl ucp team");
        }
        team->tl_ucp_team = team->team_create_req->teams[0];
        status = team->team_create_req->teams_status[0];
        ucc_free(team->team_create_req->params[0]);
        ucc_team_create_multiple_req_free(team->team_create_req);
    }
    return status;
}
