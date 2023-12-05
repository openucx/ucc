/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "tl_mlx5.h"
#include "tl_mlx5_coll.h"
#include "coll_score/ucc_coll_score.h"
#include "alltoall/alltoall.h"
#include "core/ucc_team.h"
#include <sys/shm.h>
#include "mcast/tl_mlx5_mcast.h"

static ucc_status_t ucc_tl_mlx5_topo_init(ucc_tl_mlx5_team_t *team)
{
    ucc_subset_t subset;
    ucc_status_t status;

    status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(team)->ctx_map,
                                      &UCC_TL_TEAM_MAP(team), &team->ctx_map);
    if (UCC_OK != status) {
        tl_debug(UCC_TL_TEAM_LIB(team), "failed to create ctx map");
        return status;
    }
    subset.map    = team->ctx_map;
    subset.myrank = UCC_TL_TEAM_RANK(team);

    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(team)->topo, &team->topo);

    if (UCC_OK != status) {
        tl_debug(UCC_TL_TEAM_LIB(team), "failed to init team topo");
        goto err_topo_init;
    }

    return UCC_OK;
err_topo_init:
    ucc_ep_map_destroy_nested(&team->ctx_map);
    return status;
}

void ucc_tl_mlx5_topo_cleanup(ucc_tl_mlx5_team_t *team)
{
    if (!team->topo) {
        return;
    }
    ucc_ep_map_destroy_nested(&team->ctx_map);
    ucc_topo_cleanup(team->topo);
    team->topo = NULL;
}

UCC_CLASS_INIT_FUNC(ucc_tl_mlx5_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_mlx5_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_mlx5_context_t);
    ucc_status_t status = UCC_OK;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    status = ucc_tl_mlx5_topo_init(self);
    if (status != UCC_OK) {
        tl_debug(ctx->super.super.lib, "failed to init team topo");
        return status;
    }

    self->a2a = NULL;
    status = ucc_tl_mlx5_team_init_alltoall(self);
    if (UCC_OK != status) {
        return status;
    }

    self->mcast  = NULL;
    status = ucc_tl_mlx5_mcast_team_init(tl_context, &(self->mcast), &(ctx->mcast), params,
                                         &(UCC_TL_MLX5_TEAM_LIB(self)->cfg.mcast_conf));
    if (UCC_OK != status) {
        return status;
    }

    self->state = TL_MLX5_TEAM_STATE_INIT;
    tl_debug(tl_context->lib, "posted tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);

    ucc_tl_mlx5_dm_cleanup(self);
    ucc_tl_mlx5_alltoall_cleanup(self);
    ucc_tl_mlx5_topo_cleanup(self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_mlx5_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_mlx5_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_mlx5_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t *tl_team   = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_team_t         *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_subset_t        subset    = {.map    = UCC_TL_TEAM_MAP(tl_team),
                                     .myrank = UCC_TL_TEAM_RANK(tl_team)};
    ucc_status_t        status    = UCC_OK;

    switch (tl_team->state) {
    case TL_MLX5_TEAM_STATE_INIT:
        status = ucc_service_allreduce(
            core_team, &tl_team->a2a_status.local, &tl_team->a2a_status.global,
            UCC_DT_INT32, 1, UCC_OP_MIN, subset, &tl_team->scoll_req);
        if (status < 0) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team),
                     "failed to collect global status");
            return status;
        }
        tl_team->state = TL_MLX5_TEAM_STATE_POSTED;
    case TL_MLX5_TEAM_STATE_POSTED:
        status = ucc_service_coll_test(tl_team->scoll_req);
        if (status < 0) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team),
                     "failure during service coll exchange: %s",
                     ucc_status_string(status));
            return status;
        }
        if (UCC_INPROGRESS == status) {
            return status;
        }
        ucc_service_coll_finalize(tl_team->scoll_req);
        tl_team->state = TL_MLX5_TEAM_STATE_ALLTOALL_INIT;
    case TL_MLX5_TEAM_STATE_ALLTOALL_INIT:
        tl_team->a2a_status.local =
            ucc_tl_mlx5_team_test_alltoall_start(tl_team);
        tl_team->state = TL_MLX5_TEAM_STATE_ALLTOALL_POSTED;
    case TL_MLX5_TEAM_STATE_ALLTOALL_POSTED:
        // coverity[deref_arg:FALSE]
        tl_team->a2a_status.local =
            ucc_tl_mlx5_team_test_alltoall_progress(tl_team);
        if (UCC_INPROGRESS == tl_team->a2a_status.local) {
            return UCC_INPROGRESS;
        }
        if (UCC_OK != tl_team->a2a_status.local) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team), "failed to init a2a: %s",
                     ucc_status_string(tl_team->a2a_status.local));
        }
    }

    tl_debug(team->context->lib, "initialized tl team: %p", tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_get_scores(ucc_base_team_t *  tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_mlx5_team_t *team  = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_base_context_t *ctx   = UCC_TL_TEAM_CTX(team);
    ucc_base_lib_t     *lib   = UCC_TL_TEAM_LIB(team);
    ucc_memory_type_t   mt[2] = {UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA};
    ucc_coll_score_t          *score;
    ucc_status_t               status;
    ucc_coll_score_team_info_t team_info;

    team_info.alg_fn              = NULL;
    team_info.default_score       = UCC_TL_MLX5_DEFAULT_SCORE;
    team_info.init                = ucc_tl_mlx5_coll_init;
    team_info.num_mem_types       = 2;
    team_info.supported_mem_types = mt;
    team_info.supported_colls =
        (UCC_COLL_TYPE_ALLTOALL * (team->a2a_status.local == UCC_OK)) |
        UCC_COLL_TYPE_BCAST;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

    status = ucc_coll_score_build_default(
        tl_team, UCC_TL_MLX5_DEFAULT_SCORE, ucc_tl_mlx5_coll_init,
        UCC_TL_MLX5_SUPPORTED_COLLS, mt, 2, &score);
    if (UCC_OK != status) {
        tl_debug(lib, "failed to build score map");
        return status;
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
    *score_p = NULL;
    return status;
}
