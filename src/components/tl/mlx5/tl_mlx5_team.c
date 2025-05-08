/**
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mcast/tl_mlx5_mcast_helper.h"

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

    self->global_sync_req = NULL;

    self->a2a = NULL;
    if (ctx->cfg.enable_alltoall) {
        status = ucc_tl_mlx5_team_init_alltoall(self);
        if (UCC_OK != status) {
            return status;
        }
    }

    self->mcast = NULL;

    self->local_mcast_team_ready = 0;
    if (ctx->mcast.mcast_ctx_ready) {
        status = ucc_tl_mlx5_mcast_team_init(tl_context, &(self->mcast), &(ctx->mcast),
                                             params, &(UCC_TL_MLX5_TEAM_LIB(self)->cfg.mcast_conf));
        if (UCC_OK != status) {
            tl_warn(tl_context->lib, "mcast team init failed");
        } else {
            self->local_mcast_team_ready = 1;
        }
    }

    self->mcast_state = TL_MLX5_TEAM_STATE_MCAST_CTX_CHECK;
    self->a2a_state   = TL_MLX5_TEAM_STATE_ALLTOALL_CTX_CHECK;

    tl_debug(tl_context->lib, "posted tl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);

    ucc_tl_mlx5_dm_cleanup(self);
    if (self->a2a_state != TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE) {
        ucc_tl_mlx5_alltoall_cleanup(self);
    }
    ucc_tl_mlx5_topo_cleanup(self);
    if (self->mcast_state != TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE) {
        ucc_tl_mlx5_clean_mcast_comm(self->mcast->mcast_comm);
    }
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_mlx5_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_mlx5_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_mlx5_team_t)(tl_team);
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_alltoall_team_test(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t *tl_team   = ucc_derived_of(team, ucc_tl_mlx5_team_t);

    switch (tl_team->a2a_state) {
    case TL_MLX5_TEAM_STATE_ALLTOALL_INIT:
        tl_team->a2a_status.local =
            ucc_tl_mlx5_team_test_alltoall_start(tl_team);
        tl_team->a2a_state = TL_MLX5_TEAM_STATE_ALLTOALL_POSTED;
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
            tl_team->a2a_state = TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE;
        } else {
            tl_team->a2a_state = TL_MLX5_TEAM_STATE_ALLTOALL_READY;
            tl_debug(UCC_TL_TEAM_LIB(tl_team), "initialized tl a2a team: %p",
                     tl_team);
        }
    case TL_MLX5_TEAM_STATE_ALLTOALL_READY:
    case TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE:
        return UCC_OK;
    default:
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "unknown state during a2a team: %p create", tl_team);
        return UCC_ERR_NO_RESOURCE;
    }
}

ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team      = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_context_t         *ctx          = UCC_TL_MLX5_TEAM_CTX(tl_team);
    ucc_team_t                    *core_team    = UCC_TL_CORE_TEAM(tl_team);
    ucc_subset_t                   subset       = {.map = UCC_TL_TEAM_MAP(tl_team),
                                                   .myrank = UCC_TL_TEAM_RANK(tl_team)};
    ucc_status_t                   a2a_status   = UCC_OK;
    ucc_status_t                   mcast_status = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm         = NULL;
    ucc_status_t                   status;


    if (tl_team->global_sync_req != NULL) {
        status = ucc_service_coll_test(tl_team->global_sync_req);
        if (status < 0) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team),
                     "failure during service coll exchange: %s",
                     ucc_status_string(status));
            return status;
        }
        if (UCC_INPROGRESS == status) {
            return status;
        }

        ucc_service_coll_finalize(tl_team->global_sync_req);

        tl_team->global_sync_req = NULL;

        if (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_CTX_CHECK &&
            tl_team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_CTX_CHECK ) {
            tl_team->a2a_status.global = tl_team->global_status_array[UCC_TL_MLX5_A2A_STATUS_INDEX];
            tl_team->a2a_state         = TL_MLX5_TEAM_STATE_ALLTOALL_INIT;

            if (tl_team->global_status_array[UCC_TL_MLX5_MCAST_STATUS_INDEX] != UCC_OK) {
                /* mcast context is not available for some of the team members so we cannot create
                 * mcast team */
                tl_debug(UCC_TL_TEAM_LIB(tl_team),
                         "some of the ranks do not have mcast context available so no mcast team is created");

                if (tl_team->local_mcast_team_ready) {
                    comm = tl_team->mcast->mcast_comm;
                    /* release the resources */
                    if (ibv_dereg_mr(comm->grh_mr)) {
                        tl_warn(UCC_TL_TEAM_LIB(tl_team),
                                "ibv_dereg_mr failed");
                    }
                    if (ibv_destroy_cq(comm->mcast.rcq)) {
                        tl_warn(UCC_TL_TEAM_LIB(tl_team),
                                "ibv_destroy_cq failed");
                    }
                    ucc_free(comm->params.oob);
                    ucc_free(comm);
                    ucc_free(tl_team->mcast);
                }
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE;
            } else {
                tl_debug(UCC_TL_TEAM_LIB(tl_team),
                         "all team members have mcast ctx ready");
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_INIT;
            }

            return UCC_INPROGRESS;
        } else {
            if (tl_team->global_status_array[UCC_TL_MLX5_A2A_STATUS_INDEX] != UCC_OK) {
                //a2a team not avail for some of nodes so disable it
                if (tl_team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_READY) {
                    // free the resources
                    ucc_tl_mlx5_alltoall_cleanup(tl_team);
                }
                tl_team->a2a_state = TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE;
            }

            if (tl_team->global_status_array[UCC_TL_MLX5_MCAST_STATUS_INDEX] != UCC_OK) {
                //mcast team not avail for some of nodes so disable it
                if (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_READY) {
                    // free the resources
                    ucc_tl_mlx5_clean_mcast_comm(tl_team->mcast->mcast_comm);
                }
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE;
            }

            tl_debug(team->context->lib, "team %p: MCAST component is %s ALLTOALL component is %s",
                    team, (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_READY)?"ENABLED":"DISABLED",
                    (tl_team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_READY)?"ENABLED":"DISABLED");
        }

        return UCC_OK;
    }

    ucc_assert(tl_team->global_sync_req == NULL);

    if (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_CTX_CHECK &&
        tl_team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_CTX_CHECK) {
        // check if ctx is ready for a2a and mcast
        tl_team->local_status_array[UCC_TL_MLX5_A2A_STATUS_INDEX] =
            tl_team->a2a_status.local;
        tl_team->local_status_array[UCC_TL_MLX5_MCAST_STATUS_INDEX] =
            (tl_team->local_mcast_team_ready) ? UCC_OK : UCC_ERR_NO_RESOURCE;
        goto initial_sync_post;
    }

    if (ctx->cfg.enable_alltoall) {
        a2a_status = ucc_tl_mlx5_alltoall_team_test(team);
        if (a2a_status < 0) {
            tl_warn(team->context->lib,
                    "ALLTOALL tl team: %p creation failed %d", team,
                    a2a_status);
            tl_team->a2a_state = TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE;
        }
    } else {
        tl_team->a2a_state = TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE;
    }

    if (tl_team->mcast_state != TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE) {
        mcast_status = ucc_tl_mlx5_mcast_team_test(team);
        if (mcast_status < 0) {
            tl_warn(team->context->lib, "MCAST tl team: %p creation failed %d",
                    team, mcast_status);
            tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE;
        }
    }

    if (UCC_INPROGRESS == a2a_status || UCC_INPROGRESS == mcast_status) {
        return UCC_INPROGRESS;
    }

    tl_team->local_status_array[UCC_TL_MLX5_A2A_STATUS_INDEX] =
        (tl_team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_READY) ? UCC_OK : UCC_ERR_NO_RESOURCE;
    tl_team->local_status_array[UCC_TL_MLX5_MCAST_STATUS_INDEX] =
        (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_READY) ? UCC_OK : UCC_ERR_NO_RESOURCE;

    tl_debug(UCC_TL_TEAM_LIB(tl_team),
             "posting global status, local status: ALLTOALL %d MCAST %d",
             (tl_team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_READY),
             (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_READY));

initial_sync_post:
    status = ucc_service_allreduce(
        core_team, tl_team->local_status_array, tl_team->global_status_array,
        UCC_DT_INT32, UCC_TL_MLX5_FEATURES_COUNT, UCC_OP_MIN, subset, &tl_team->global_sync_req);
    if (status < 0) {
        tl_debug(UCC_TL_TEAM_LIB(tl_team),
                 "failed to collect global status");
        return status;
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_mlx5_team_get_scores(ucc_base_team_t *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_mlx5_team_t    *team   = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_base_context_t    *ctx    = UCC_TL_TEAM_CTX(team);
    ucc_tl_mlx5_context_t *tl_ctx = ucc_derived_of(ctx, ucc_tl_mlx5_context_t);
    ucc_base_lib_t        *lib    = UCC_TL_TEAM_LIB(team);
    ucc_memory_type_t      mt[2]  = {UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA};
    ucc_coll_score_t          *score;
    ucc_status_t               status;
    ucc_coll_score_team_info_t team_info;


    team_info.alg_fn              = NULL;
    team_info.default_score       = UCC_TL_MLX5_DEFAULT_SCORE;
    team_info.init                = ucc_tl_mlx5_coll_init;
    team_info.num_mem_types       = tl_ctx->supported_mem_types & UCC_BIT(UCC_MEMORY_TYPE_CUDA) ? 2 : 1;
    team_info.supported_mem_types = mt;
    team_info.supported_colls =
        (UCC_COLL_TYPE_ALLTOALL * (team->a2a_state == TL_MLX5_TEAM_STATE_ALLTOALL_READY)) |
        (UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_ALLGATHER) *
        (team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_READY);
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
