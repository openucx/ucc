/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_doca_urom.h"
#include "utils/ucc_malloc.h"
#include "core/ucc_team.h"

UCC_CLASS_INIT_FUNC(ucc_cl_doca_urom_team_t, ucc_base_context_t *cl_context,
                    const ucc_base_team_params_t                *params)
{
    union doca_data             cookie = {0};
    doca_error_t                result = DOCA_SUCCESS;
    ucc_cl_doca_urom_context_t *ctx    =
        ucc_derived_of(cl_context, ucc_cl_doca_urom_context_t);
    ucc_status_t                status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super, params);

    self->teams = (ucc_team_h **) ucc_malloc(
                        sizeof(ucc_team_h *) * UCC_CL_DOCA_UROM_MAX_TEAMS);

    if (!self->teams) {
        cl_error(cl_context->lib,
                 "failed to allocate %zd bytes for doca_urom teams",
                 sizeof(ucc_team_h *) * UCC_CL_DOCA_UROM_MAX_TEAMS);
        status = UCC_ERR_NO_MEMORY;
        return status;
    }

    self->n_teams = 0;
    self->score_map = NULL;

    cookie.ptr = &self->res;

    /* Send the command to create a team on the DPU */
    result = ucc_cl_doca_urom_task_team_create(
                ctx->urom_ctx.urom_worker,
                cookie,
                ctx->urom_ctx.ctx_rank,
                0 /* start */,
                1 /* stride */,
                params->params.oob.n_oob_eps /* size */,
                ctx->urom_ctx.urom_ucc_context,
                ucc_cl_doca_urom_team_create_finished);

    if (result != DOCA_SUCCESS) {
        cl_error(cl_context->lib, "failed to create UCC team task");
        return UCC_ERR_NO_RESOURCE;
    }

    self->res.team_create.status = UCC_INPROGRESS;

    cl_debug(cl_context->lib, "posted cl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_doca_urom_team_t)
{
    cl_debug(self->super.super.context->lib, "finalizing cl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_doca_urom_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_cl_doca_urom_team_t, ucc_cl_team_t);

ucc_status_t ucc_cl_doca_urom_team_destroy(ucc_base_team_t *cl_team)
{
    return UCC_OK;
}

ucc_status_t ucc_cl_doca_urom_team_create_test(ucc_base_team_t *cl_team)
{
    ucc_cl_doca_urom_team_t                    *team         =
        ucc_derived_of(cl_team, ucc_cl_doca_urom_team_t);
    ucc_cl_doca_urom_context_t                 *ctx          =
        UCC_CL_DOCA_UROM_TEAM_CTX(team);
    ucc_memory_type_t                           mem_types[2] =
        {UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA};
    struct ucc_cl_doca_urom_result             *res          = &team->res;
    struct ucc_cl_doca_urom_team_create_result *team_create  =
        &res->team_create;
    ucc_coll_score_t                           *score        = NULL;
    int                                         mt_n         = 2;
    ucc_status_t                                ucc_status;
    int                                         ret;

    ret = doca_pe_progress(ctx->urom_ctx.urom_pe);
    if (ret == 0 && res->result == DOCA_SUCCESS) {
        return UCC_INPROGRESS;
    }

    if (res->result != DOCA_SUCCESS) {
        cl_error(ctx->super.super.lib,
                 "UCC team create task failed: DOCA status %d", res->result);
        return UCC_ERR_NO_MESSAGE;
    }

    if (team_create->status == UCC_OK) {
        team->teams[team->n_teams] = team_create->team;
        ++team->n_teams;
        ucc_status = ucc_coll_score_build_default(
                        cl_team, UCC_CL_DOCA_UROM_DEFAULT_SCORE,
                        ucc_cl_doca_urom_coll_init,
                        UCC_COLL_TYPE_ALLREDUCE,
                        mem_types, mt_n, &score);
        if (UCC_OK != ucc_status) {
            return ucc_status;
        }

        ucc_status = ucc_coll_score_build_map(score, &team->score_map);
        if (UCC_OK != ucc_status) {
            cl_error(ctx->super.super.lib, "failed to build score map");
        }
        team->score = score;
        ucc_coll_score_set(team->score, UCC_CL_DOCA_UROM_DEFAULT_SCORE);

        return UCC_OK;
    } else if (team_create->status < 0) {
        cl_error(ctx->super.super.lib, "failed to create team: %s",
                ucc_status_string(team_create->status));
        return team_create->status;
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_cl_doca_urom_team_get_scores(ucc_base_team_t   *cl_team,
                                          ucc_coll_score_t **score)
{
    ucc_cl_doca_urom_team_t   *team = ucc_derived_of(cl_team,
                                                   ucc_cl_doca_urom_team_t);
    ucc_base_context_t        *ctx  = UCC_CL_TEAM_CTX(team);
    ucc_coll_score_team_info_t team_info;
    ucc_status_t               status;

    status = ucc_coll_score_dup(team->score, score);
    if (UCC_OK != status) {
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        team_info.alg_fn              = NULL;
        team_info.default_score       = UCC_CL_DOCA_UROM_DEFAULT_SCORE;
        team_info.init                = NULL;
        team_info.num_mem_types       = 0;
        team_info.supported_mem_types = NULL; /* all memory types supported*/
        team_info.supported_colls     = UCC_COLL_TYPE_ALLREDUCE;
        team_info.size                = UCC_CL_TEAM_SIZE(team);

        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                &team->super.super, *score);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }

    return UCC_OK;

err:
    ucc_coll_score_free(*score);
    *score = NULL;
    return status;
}
