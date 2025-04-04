/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"
#include "core/ucc_team.h"
#include <sharp/api/version.h>

UCC_CLASS_INIT_FUNC(ucc_tl_sharp_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_sharp_context_t          *ctx =
        ucc_derived_of(tl_context, ucc_tl_sharp_context_t);
    struct sharp_coll_context       *sharp_ctx = ctx->sharp_context;
    struct sharp_coll_comm_init_spec comm_spec;
    int                              ret;
    ucc_status_t                     status;
    ucc_subset_t                     set;

    if (!(params->params.mask & UCC_TEAM_PARAM_FIELD_OOB)) {
        tl_debug(ctx->super.super.lib, "team OOB required for sharp team");
        return UCC_ERR_INVALID_PARAM;
    }

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    self->sharp_context = NULL;
    self->rcache        = NULL;
    self->oob_ctx.ctx   = UCC_TL_TEAM_CTX(self);

    set.myrank = UCC_TL_TEAM_RANK(self);
    set.map    = UCC_TL_TEAM_MAP(self);

    if (UCC_TL_SHARP_TEAM_LIB(self)->cfg.use_internal_oob) {
        status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(self)->ctx_map,
                                 &UCC_TL_TEAM_MAP(self),
                                 &self->oob_ctx.subset.map);
        if (status != UCC_OK) {
            return status;
        }
        self->oob_ctx.subset.myrank = UCC_TL_TEAM_RANK(self);
    } else {
        self->oob_ctx.oob = &UCC_TL_TEAM_OOB(self);
    }

    status = ucc_topo_init(set, ctx->super.super.ucc_context->topo, &self->topo);
    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to init team topo");
        if (UCC_TL_SHARP_TEAM_LIB(self)->cfg.use_internal_oob) {
            ucc_ep_map_destroy_nested(&self->oob_ctx.subset.map);
        }
        return status;
    }

    if (ucc_topo_max_ppn(self->topo) > ctx->cfg.team_max_ppn) {
        tl_debug(ctx->super.super.lib, "sharp team not supported with ppn > 1");
        status = UCC_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    if (sharp_ctx == NULL) {
        status = ucc_tl_sharp_context_init(ctx, &self->sharp_context,
                                           &self->oob_ctx, self->topo);
        if (status != UCC_OK) {
            goto cleanup;
        }

        if (ctx->cfg.use_rcache) {
            status = ucc_tl_sharp_rcache_create(self->sharp_context, &self->rcache);
            if (status != UCC_OK) {
                tl_error(ctx->super.super.lib, "failed to create rcache");
                goto cleanup;
            }
        }

        status = ucc_context_progress_register(
                tl_context->ucc_context, (ucc_context_progress_fn_t)sharp_coll_progress,
                self->sharp_context);
        if (status != UCC_OK) {
            tl_error(ctx->super.super.lib, "failed to register progress function");
            goto cleanup;
        }

        sharp_ctx = self->sharp_context;
    } else {
        self->sharp_context = sharp_ctx;
        self->rcache        = ctx->rcache;
    }

#if SHARP_API > SHARP_VERSION(3, 0)
    if ((ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] ==
         SHARP_DTYPE_UNKNOWN) ||
        (ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(SHARP_DTYPE_UINT8)] ==
         SHARP_DTYPE_UNKNOWN) ||
        (ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(SHARP_DTYPE_BFLOAT16)] ==
         SHARP_DTYPE_UNKNOWN)) {

        if (ctx->sharp_caps.support_mask.dtypes & UCC_BIT(SHARP_DTYPE_INT8)) {
            tl_debug(ctx->super.super.lib, "enabling support for UCC_DT_INT8");
            ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] = SHARP_DTYPE_INT8;
        } else {
            tl_debug(ctx->super.super.lib, "disabling support for UCC_DT_INT8");
            ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] = SHARP_DTYPE_NULL;
        }

        if (ctx->sharp_caps.support_mask.dtypes & UCC_BIT(SHARP_DTYPE_UINT8)) {
            tl_debug(ctx->super.super.lib, "enabling support for UCC_DT_UINT8");
            ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)] = SHARP_DTYPE_UINT8;
        } else {
            tl_debug(ctx->super.super.lib, "disabling support for UCC_DT_UINT8");
            ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)] = SHARP_DTYPE_NULL;
        }


        if (ctx->sharp_caps.support_mask.dtypes & UCC_BIT(SHARP_DTYPE_BFLOAT16)) {
            tl_debug(ctx->super.super.lib, "enabling support for UCC_DT_BFLOAT16");
            ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)] = SHARP_DTYPE_BFLOAT16;
        } else {
            tl_debug(ctx->super.super.lib, "disabling support for UCC_DT_BFLOAT16");
            ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)] = SHARP_DTYPE_NULL;
        }
    }
#endif

    comm_spec.rank              = UCC_TL_TEAM_RANK(self);
    comm_spec.size              = UCC_TL_TEAM_SIZE(self);
    comm_spec.group_world_ranks = NULL;
    comm_spec.oob_ctx           = &self->oob_ctx;

    ret = sharp_coll_comm_init(sharp_ctx,
                               &comm_spec, &self->sharp_comm);
    if (ret < 0) {
        tl_debug(ctx->super.super.lib, "sharp group create failed:%s(%d)",
                 sharp_coll_strerror(ret), ret);
        status = UCC_ERR_NO_RESOURCE;
        goto cleanup;
    }

    tl_debug(self->super.super.context->lib,
             "initialized tl team: %p size:%d", self, UCC_TL_TEAM_SIZE(self));
    return UCC_OK;
cleanup:
    if (ctx->cfg.context_per_team) {
        if (self->rcache) {
            ucc_rcache_destroy(self->rcache);
        }
        if (self->sharp_context) {
            ucc_context_progress_deregister(
                    tl_context->ucc_context,
                    (ucc_context_progress_fn_t)sharp_coll_progress,
                    self->sharp_context);
            sharp_coll_finalize(self->sharp_context);
        }
    }
    ucc_topo_cleanup(self->topo);
    if (UCC_TL_SHARP_TEAM_LIB(self)->cfg.use_internal_oob) {
        ucc_ep_map_destroy_nested(&self->oob_ctx.subset.map);
    }
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_sharp_team_t)
{
    ucc_tl_sharp_context_t *ctx = ucc_derived_of(UCC_TL_TEAM_CTX(self), ucc_tl_sharp_context_t);

    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
    sharp_coll_comm_destroy(self->sharp_comm);
    ucc_topo_cleanup(self->topo);

    if (ctx->cfg.context_per_team) {
        if (UCC_TL_SHARP_TEAM_LIB(self)->cfg.use_internal_oob) {
            if (self->rcache != NULL) {
                ucc_rcache_destroy(self->rcache);
            }
            if (self->sharp_context != NULL) {
                ucc_context_progress_deregister(
                        self->super.super.context->ucc_context,
                        (ucc_context_progress_fn_t)sharp_coll_progress, self->sharp_context);
                sharp_coll_finalize(self->sharp_context);
            }
        }
    }
    if (UCC_TL_SHARP_TEAM_LIB(self)->cfg.use_internal_oob) {
        ucc_ep_map_destroy_nested(&self->oob_ctx.subset.map);
    }
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_sharp_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_sharp_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_sharp_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_sharp_team_t)(tl_team);
    return UCC_OK;
}

/* sharp team create is blocking, return UCC_OK always */
ucc_status_t ucc_tl_sharp_team_create_test(ucc_base_team_t *tl_team) //NOLINT
{
    return UCC_OK;
}

static ucc_status_t ucc_tl_sharp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);

    tl_debug(UCC_TASK_LIB(task), "finalizing coll task %p", task);
    UCC_TL_SHARP_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
    return UCC_OK;
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
#if HAVE_DECL_SHARP_COLL_DO_REDUCE_SCATTER
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        status = ucc_tl_sharp_reduce_scatter_init(task);
        break;
#endif
#if HAVE_DECL_SHARP_COLL_DO_ALLGATHER
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_sharp_allgather_init(task);
        break;
#endif
    default:
        tl_debug(UCC_TASK_LIB(task),
                 "collective %d is not supported by sharp tl",
                 coll_args->args.coll_type);
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        goto free_task;
    }

    tl_debug(UCC_TASK_LIB(task), "init coll task %p", task);
    *task_h = &task->super;
    return status;

free_task:
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_sharp_team_get_scores(ucc_base_team_t  *tl_team,
                                          ucc_coll_score_t **score_p)
{
    ucc_tl_sharp_team_t *team = ucc_derived_of(tl_team, ucc_tl_sharp_team_t);
    ucc_base_context_t  *ctx  = UCC_TL_TEAM_CTX(team);
    ucc_coll_score_t    *score;
    ucc_status_t         status;
    ucc_coll_score_team_info_t team_info;
    int i;

    team_info.alg_fn              = NULL;
    team_info.default_score       = UCC_TL_SHARP_DEFAULT_SCORE;
    team_info.init                = ucc_tl_sharp_coll_init;
    team_info.num_mem_types       = 0;
    team_info.supported_mem_types = NULL; /* all memory types supported*/
    team_info.supported_colls     = UCC_TL_SHARP_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);
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

    for (i = 0; i < UCC_TL_SHARP_N_DEFAULT_ALG_SELECT_STR; i++) {
        status = ucc_coll_score_update_from_str(
            ucc_tl_sharp_default_alg_select_str[i], &team_info,
            &team->super.super, score);
        if (UCC_OK != status) {
            tl_error(tl_team->context->lib,
                     "failed to apply default coll select setting: %s",
                     ucc_tl_sharp_default_alg_select_str[i]);
            goto err;
        }
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                &team->super.super, score);
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
