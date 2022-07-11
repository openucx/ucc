/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"

UCC_CLASS_INIT_FUNC(ucc_tl_sharp_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_sharp_context_t         *ctx =
        ucc_derived_of(tl_context, ucc_tl_sharp_context_t);
    struct sharp_coll_comm_init_spec comm_spec;
    int                              ret;

    if (!(params->params.mask & UCC_TEAM_PARAM_FIELD_OOB)) {
        tl_debug(ctx->super.super.lib, "team OOB required for sharp team");
        return UCC_ERR_INVALID_PARAM;
    }

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    self->oob_ctx.ctx = UCC_TL_TEAM_CTX(self);
    self->oob_ctx.oob = &UCC_TL_TEAM_OOB(self);

    comm_spec.rank              = UCC_TL_TEAM_RANK(self);
    comm_spec.size              = UCC_TL_TEAM_SIZE(self);
    comm_spec.group_world_ranks = NULL;
    comm_spec.oob_ctx           = &self->oob_ctx;

    ret = sharp_coll_comm_init(ctx->sharp_context,
                               &comm_spec, &self->sharp_comm);
    if (ret < 0) {
        tl_error(ctx->super.super.lib,
                "sharp group create failed:%s(%d)",
                sharp_coll_strerror(ret), ret);
        return UCC_ERR_NO_RESOURCE;
    }

    tl_info(self->super.super.context->lib, "initialized tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_sharp_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
    sharp_coll_comm_destroy(self->sharp_comm);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_sharp_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_sharp_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_sharp_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_sharp_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_team_create_test(ucc_base_team_t *tl_team)
{
    return UCC_OK;
}

static ucc_status_t ucc_tl_sharp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);

    tl_info(UCC_TASK_LIB(task), "finalizing coll task %p", task);
    UCC_TL_SHARP_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                         ucc_coll_task_t *coll_task)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_sharp_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task_h)
{
    ucc_tl_sharp_context_t *sharp_ctx  = ucc_derived_of(team->context,
                                                      ucc_tl_sharp_context_t);
    ucc_tl_sharp_task_t *task;
    ucc_status_t         status;

    task = ucc_mpool_get(&sharp_ctx->req_mp);
    ucc_coll_task_init(&task->super, coll_args, team);
    UCC_TL_SHARP_PROFILE_REQUEST_NEW(task, "tl_sharp_task", 0);

    task->req_handle           = NULL;
    task->super.finalize       = ucc_tl_sharp_coll_finalize;
    task->super.triggered_post = ucc_tl_sharp_triggered_post;

    switch (coll_args->args.coll_type)
    {
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_sharp_allreduce_init(task);
        break;
    case UCC_COLL_TYPE_BARRIER:
        status = ucc_tl_sharp_barrier_init(task);
        break;
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_sharp_bcast_init(task);
        break;
    default:
        tl_error(UCC_TASK_LIB(task),
                 "collective %d is not supported by sharp tl",
                 coll_args->args.coll_type);
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        goto free_task;
    }

    tl_info(UCC_TASK_LIB(task), "init coll task %p", task);
    *task_h = &task->super;
    return status;

free_task:
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_sharp_team_get_scores(ucc_base_team_t   *tl_team,
                                          ucc_coll_score_t **score_p)
{
    ucc_tl_sharp_team_t *team = ucc_derived_of(tl_team, ucc_tl_sharp_team_t);
    ucc_base_context_t  *ctx  = UCC_TL_TEAM_CTX(team);
    ucc_coll_score_t    *score;
    ucc_status_t         status;

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status =
        ucc_coll_score_build_default(tl_team, UCC_TL_SHARP_DEFAULT_SCORE,
                                     ucc_tl_sharp_coll_init,
                                     UCC_TL_SHARP_SUPPORTED_COLLS,
                                     NULL, 0, &score);
    if (UCC_OK != status) {
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team),
            ucc_tl_sharp_coll_init, &team->super.super,
            UCC_TL_SHARP_DEFAULT_SCORE, NULL);
        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }

    *score_p = score;
    return UCC_OK;
err:
    ucc_coll_score_free(score);
    return status;
}
