/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "coll_score/ucc_coll_score.h"
#include "core/ucc_team.h"
#include "tl_self.h"

UCC_CLASS_INIT_FUNC(ucc_tl_self_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_self_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_self_context_t);

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    tl_debug(tl_context->lib, "posted tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_self_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_self_team_t, ucc_base_team_t);

UCC_CLASS_DEFINE(ucc_tl_self_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_self_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_self_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_self_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_self_team_t *team = ucc_derived_of(tl_team, ucc_tl_self_team_t);

    tl_debug(tl_team->context->lib, "initialized tl team: %p", team);
    return UCC_OK;
}

ucc_status_t ucc_tl_self_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_self_team_t *team = ucc_derived_of(tl_team, ucc_tl_self_team_t);
    ucc_base_context_t *ctx  = UCC_TL_TEAM_CTX(team);
    int                 mt_n = 0, i;
    ucc_memory_type_t   mem_types[UCC_MEMORY_TYPE_LAST];
    ucc_coll_score_t   *score;
    ucc_status_t        status;
    ucc_coll_score_team_info_t team_info;

    for (i = 0; i < UCC_MEMORY_TYPE_LAST; i++) {
        mem_types[mt_n++] = (ucc_memory_type_t)i;
    }

    team_info.alg_fn              = NULL;
    team_info.default_score       = UCC_TL_SELF_DEFAULT_SCORE;
    team_info.init                = ucc_tl_self_coll_init;
    team_info.num_mem_types       = mt_n;
    team_info.supported_mem_types = mem_types; /* all memory types supported*/
    team_info.supported_colls     = UCC_TL_SELF_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

    status = ucc_coll_score_build_default(
        tl_team, UCC_TL_SELF_DEFAULT_SCORE, ucc_tl_self_coll_init,
        UCC_TL_SELF_SUPPORTED_COLLS, mem_types, mt_n, &score);

    if (UCC_OK != status) {
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                &team->super.super, score);
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
