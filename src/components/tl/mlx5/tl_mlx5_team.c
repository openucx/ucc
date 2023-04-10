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

static ucc_mpool_ops_t ucc_tl_mlx5_dm_ops;
static ucc_status_t    ucc_tl_mlx5_dm_init(ucc_tl_mlx5_team_t *team);
static void            ucc_tl_mlx5_dm_cleanup(ucc_tl_mlx5_team_t *team);

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

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score_t");
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team), NULL, tl_team,
            UCC_TL_MLX5_DEFAULT_SCORE, NULL, &mt, 1);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }
    *score_p = score;
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
