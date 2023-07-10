/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "tl_mlx5.h"
#include "tl_mlx5_dm.h"
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
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create ctx map");
        return status;
    }
    subset.map    = team->ctx_map;
    subset.myrank = UCC_TL_TEAM_RANK(team);

    status = ucc_topo_init(subset, UCC_TL_CORE_CTX(team)->topo, &team->topo);

    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init team topo");
        goto err_topo_init;
    }

    return UCC_OK;
err_topo_init:
    ucc_ep_map_destroy_nested(&team->ctx_map);
    return status;
}

static void ucc_tl_mlx5_topo_cleanup(ucc_tl_mlx5_team_t *team)
{
    ucc_ep_map_destroy_nested(&team->ctx_map);
    ucc_topo_cleanup(team->topo);
}

UCC_CLASS_INIT_FUNC(ucc_tl_mlx5_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_mlx5_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_mlx5_context_t);
    ucc_status_t status = UCC_OK;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    self->a2a    = NULL;
    self->dm_ptr = NULL;

    status = ucc_tl_mlx5_topo_init(self);
    if (status != UCC_OK) {
        tl_error(ctx->super.super.lib, "failed to init team topo");
        return status;
    }

    if (ucc_topo_get_sbgp(self->topo, UCC_SBGP_NODE)->group_rank ==
        MLX5_ASR_RANK) {
        status = ucc_tl_mlx5_dm_init(self);
        if (UCC_OK != status) {
            tl_debug(UCC_TL_TEAM_LIB(self), "failed to init device memory");
        }
    }

    self->status[0] = status;
    self->state     = TL_MLX5_TEAM_STATE_INIT;

    self->mcast  = NULL;
    status = ucc_tl_mlx5_mcast_team_init(tl_context, &(self->mcast), &(ctx->mcast), params,
                                         &(UCC_TL_MLX5_TEAM_LIB(self)->cfg.mcast_conf));
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    tl_debug(tl_context->lib, "posted tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);

    ucc_tl_mlx5_alltoall_cleanup(self);
    ucc_tl_mlx5_dm_cleanup(self);
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
    ucc_subset_t        subset    = {.map.type   = UCC_EP_MAP_FULL,
                                     .map.ep_num = core_team->size,
                                     .myrank     = core_team->rank};
    ucc_status_t        status    = UCC_OK;

    switch (tl_team->state) {
    case TL_MLX5_TEAM_STATE_INIT:
        status = ucc_service_allreduce(
                    core_team, &tl_team->status[0], &tl_team->status[1],
                    UCC_DT_INT32, 1, UCC_OP_MIN, subset, &tl_team->scoll_req);
        if (status < 0) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "failed to collect global status");
            return status;
        }
        tl_team->state = TL_MLX5_TEAM_STATE_POSTED;
    case TL_MLX5_TEAM_STATE_POSTED:
        status = ucc_service_coll_test(tl_team->scoll_req);
        if (status < 0) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "failure during service coll exchange: %s",
                     ucc_status_string(status));
            return status;
        }
        if (UCC_INPROGRESS == status) {
            return status;
        }
        ucc_assert(status == UCC_OK);
        ucc_service_coll_finalize(tl_team->scoll_req);
        if (tl_team->status[1] != UCC_OK) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team),
                     "node leader failed during device memory init: %s",
                     ucc_status_string(tl_team->status[1]));
            ucc_tl_mlx5_team_destroy(team);
            return tl_team->status[1];
        }
        tl_team->state = TL_MLX5_TEAM_STATE_ALLTOALL_INIT;
    case TL_MLX5_TEAM_STATE_ALLTOALL_INIT:
        status = ucc_tl_mlx5_team_alltoall_init_start(tl_team);
        if (status != UCC_OK) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team), "failed to init a2a: %s",
                     ucc_status_string(status));
            return status;
        }
        tl_team->state = TL_MLX5_TEAM_STATE_ALLTOALL_POSTED;
    case TL_MLX5_TEAM_STATE_ALLTOALL_POSTED:
        status = ucc_tl_mlx5_team_alltoall_init_progress(tl_team);
    }
    if (status < 0) {
        tl_debug(team->context->lib, "failed creating tl team: %p", tl_team);
    } else if (status == UCC_OK) {
        tl_debug(team->context->lib, "initialized tl team: %p", tl_team);
    }
    return status;
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
    team_info.init                = NULL;
    team_info.num_mem_types       = 2;
    team_info.supported_mem_types = mt;
    team_info.supported_colls     = UCC_TL_MLX5_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score_t");
        return status;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_HOST, 0,
        MAX_MSG_SIZE * UCC_TL_TEAM_SIZE(team), UCC_TL_MLX5_DEFAULT_SCORE,
        ucc_tl_mlx5_coll_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "failed to add range to score_t");
        return status;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_CUDA, 0,
        MAX_MSG_SIZE * UCC_TL_TEAM_SIZE(team), UCC_TL_MLX5_DEFAULT_SCORE,
        ucc_tl_mlx5_coll_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "failed to add range to score_t");
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
