/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "tl_mlx5_dm.h"
#include "coll_score/ucc_coll_score.h"
#include "alltoall/alltoall.h"
#include "core/ucc_team.h"
#include <sys/shm.h>

static ucc_status_t ucc_tl_mlx5_topo_init(ucc_tl_mlx5_team_t *team)
{
    ucc_subset_t  subset;
    ucc_status_t  status;

    status = ucc_ep_map_create_nested(&UCC_TL_CORE_TEAM(team)->ctx_map,
                                      &UCC_TL_TEAM_MAP(team),
                                      &team->ctx_map);
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

static void ucc_tl_mlx5_topo_destroy(ucc_tl_mlx5_team_t *team)
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

    if (ucc_topo_get_sbgp(UCC_TL_CORE_TEAM(self)->topo, UCC_SBGP_NODE)
            ->group_rank == 0) {
        /* MEMIC alloc, todo move to CTX and share on node*/
        status = ucc_tl_mlx5_dm_init(self);
        if (UCC_OK != status) {
            tl_error(UCC_TL_TEAM_LIB(self),
                        "failed to init device memory");
            return status;
        }
    }

    status = ucc_tl_mlx5_topo_init(self); 
    if (status != UCC_OK){
        tl_error(ctx->super.super.lib, "failed to init team topo");
        return status;
    }

    status = ucc_tl_mlx5_a2a_init_start(self);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);

    ucc_tl_mlx5_a2a_cleanup(self);
    ucc_tl_mlx5_dm_cleanup(self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_mlx5_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_mlx5_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_mlx5_team_t *team = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);

    ucc_tl_mlx5_topo_destroy(team);
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_mlx5_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args, /* NOLINT */
                                   ucc_base_team_t *     team,      /* NOLINT */
                                   ucc_coll_task_t **    task)      /* NOLINT */
{
    return UCC_OK;
}
ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_mlx5_team_t *team = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_status_t        status;

    status = ucc_tl_mlx5_a2a_init_progress(team);

    if (status == UCC_OK) {
        tl_debug(tl_team->context->lib, "initialized tl team: %p", team);
    }
    return status;
}
