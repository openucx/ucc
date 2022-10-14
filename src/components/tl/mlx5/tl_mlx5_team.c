/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
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
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_mlx5_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_mlx5_team_t *team = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_status_t        status;

    status = ucc_tl_mlx5_a2a_init_progress(team);

    if (status == UCC_OK) {
        tl_info(tl_team->context->lib, "initialized tl team: %p", team);

    }
    return status;
}

ucc_status_t ucc_tl_mlx5_team_get_scores(ucc_base_team_t *  tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_mlx5_team_t *team = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_base_context_t *ctx  = UCC_TL_TEAM_CTX(team);
    ucc_base_lib_t *    lib  = UCC_TL_TEAM_LIB(team);
    ucc_coll_score_t *  score;
    ucc_status_t        status;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score_t");
        return status;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_HOST, 0,
        MAX_MSG_SIZE * UCC_TL_TEAM_SIZE(team), UCC_TL_MLX5_DEFAULT_SCORE,
        ucc_tl_mlx5_alltoall_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        return status;
    }

    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_CUDA, 0,
        MAX_MSG_SIZE * UCC_TL_TEAM_SIZE(team), UCC_TL_MLX5_DEFAULT_SCORE,
        ucc_tl_mlx5_alltoall_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team), NULL, tl_team,
            UCC_TL_MLX5_DEFAULT_SCORE, NULL);

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

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task)
{
    return UCC_OK;
}

static void ucc_tl_mlx5_dm_cleanup(ucc_tl_mlx5_team_t *team)
{
    if (!team->dm_ptr) {
        return;
    }
    ibv_dereg_mr(team->dm_mr);
    if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host) {
        ucc_free(team->dm_ptr);
    } else {
        ibv_free_dm(team->dm_ptr);
    }
    ucc_mpool_cleanup(&team->dm_pool, 1);
}

static ucc_status_t ucc_tl_mlx5_dm_init(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    size_t memic_chunk         = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_buf_size;
    size_t n_memic_chunks      = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_buf_num;
    int    dm_host             = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host;
    struct ibv_device_attr_ex attr;
    struct ibv_alloc_dm_attr  dm_attr;
    int                       max_n_chunks, chunks_to_alloc, i;
    ucc_status_t              status;

    if (dm_host) {
        max_n_chunks    = 8;
        chunks_to_alloc = (n_memic_chunks == UCC_ULUNITS_AUTO) ? max_n_chunks
            : n_memic_chunks;
        dm_attr.length  = chunks_to_alloc * memic_chunk;
        team->dm_ptr    = ucc_malloc(dm_attr.length, "memic_host");
        if (!team->dm_ptr) {
            tl_error(UCC_TL_TEAM_LIB(team), " memic_host allocation failed");
            return UCC_ERR_NO_MEMORY;
        }
        team->dm_mr =
            ibv_reg_mr(ctx->shared_pd, team->dm_ptr, dm_attr.length,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    } else {
        attr.comp_mask = 0;
        if (ibv_query_device_ex(ctx->ib_ctx, NULL, &attr)) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to query device (errno=%d)",
                     errno);
            return UCC_ERR_NO_MESSAGE;
        }
        if (!attr.max_dm_size) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "device doesn't support dm allocation");
            return UCC_ERR_NO_MESSAGE;
        }
        memset(&dm_attr, 0, sizeof(dm_attr));
        max_n_chunks = attr.max_dm_size / memic_chunk;
        max_n_chunks--;
        chunks_to_alloc = (n_memic_chunks == UCC_ULUNITS_AUTO) ? max_n_chunks
            : n_memic_chunks;

        for (i = chunks_to_alloc; i > 0; i--) {
            dm_attr.length = i * memic_chunk;
            team->dm_ptr   = ibv_alloc_dm(ctx->ib_ctx, &dm_attr);
            if (team->dm_ptr) {
                break;
            }
        }
        if (!team->dm_ptr) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "dev mem allocation failed, attr.max %zd, errno %d",
                     attr.max_dm_size, errno);
            return UCC_ERR_NO_MESSAGE;
        }
        if (n_memic_chunks != UCC_ULUNITS_AUTO && i != n_memic_chunks) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "couldn't allocate memic chunks, required %zd allocated "
                     "%d, max %d",
                     n_memic_chunks, i, max_n_chunks);
            return UCC_ERR_NO_MESSAGE;
        }
        n_memic_chunks = i;
        team->dm_mr =
            ibv_reg_dm_mr(ctx->shared_pd, team->dm_ptr, 0, dm_attr.length,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_ZERO_BASED);
    }
    if (!team->dm_mr) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to reg memic");
        return UCC_ERR_NO_MESSAGE;
    }

    team->oob_req = NULL;
    status = ucc_mpool_init(&team->dm_pool, 0, sizeof(ucc_tl_mlx5_dm_chunk_t),
                            0, UCC_CACHE_LINE_SIZE, n_memic_chunks,
                            n_memic_chunks, &ucc_tl_mlx5_dm_ops,
                            UCC_THREAD_MULTIPLE, "mlx5 dm pool");
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init dm pool");
        return status;
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_dm_chunk_alloc(ucc_mpool_t *mp, //NOLINT
                                               size_t *size_p, void **chunk_p)
{
    *chunk_p = ucc_malloc(*size_p, "mlx5 dm");
    if (!*chunk_p) {
        return UCC_ERR_NO_MEMORY;
    }
    return UCC_OK;
}

static void ucc_tl_mlx5_dm_chunk_init(ucc_mpool_t *mp,        //NOLINT
                                      void *obj, void *chunk) //NOLINT
{
    ucc_tl_mlx5_dm_chunk_t *c = (ucc_tl_mlx5_dm_chunk_t *)obj;
    ucc_tl_mlx5_team_t *    team =
        ucc_container_of(mp, ucc_tl_mlx5_team_t, dm_pool);
    const size_t memic_chunk = 8192;

    c->offset                = (ptrdiff_t)team->oob_req;
    team->oob_req            = PTR_OFFSET(team->oob_req, memic_chunk);
}

static void ucc_tl_mlx5_dm_chunk_release(ucc_mpool_t *mp, void *chunk) //NOLINT
{
    ucc_free(chunk);
}

static ucc_mpool_ops_t ucc_tl_mlx5_dm_ops = {
    .chunk_alloc   = ucc_tl_mlx5_dm_chunk_alloc,
    .chunk_release = ucc_tl_mlx5_dm_chunk_release,
    .obj_init      = ucc_tl_mlx5_dm_chunk_init,
    .obj_cleanup   = NULL
};
