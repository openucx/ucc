/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_dm.h"
#include "alltoall/alltoall.h"

#define DM_HOST_AUTO_NUM_CHUNKS 8

static void ucc_tl_mlx5_alltoall_atomic_free(ucc_tl_mlx5_team_t *team)
{
    if (!team->a2a || !team->a2a->net.atomic.counters) {
        return;
    }

    ibv_dereg_mr(team->a2a->net.atomic.mr);
#if ATOMIC_IN_MEMIC
    ibv_free_dm(team->a2a->net.atomic.counters);
#else
    ucc_free(team->a2a->net.atomic.counters);
#endif
    team->a2a->net.atomic.counters = NULL;
}

static ucc_status_t ucc_tl_mlx5_alltoall_atomic_alloc(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t  *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_alltoall_t *a2a = team->a2a;
    size_t                  size;

    size = sizeof(*a2a->net.atomic.counters) * MAX_OUTSTANDING_OPS;
#if ATOMIC_IN_MEMIC
    struct ibv_alloc_dm_attr dm_attr;
    memset(&dm_attr, 0, sizeof(dm_attr));
    dm_attr.length           = size;
    a2a->net.atomic.counters = ibv_alloc_dm(ctx->shared_ctx, &dm_attr);
#else
    a2a->net.atomic.counters = ucc_malloc(size, "atomic");
#endif

    if (!a2a->net.atomic.counters) {
        tl_debug(UCC_TL_TEAM_LIB(team),
                 "failed to allocate %zd bytes for atomic counters array",
                 size);
        return UCC_ERR_NO_MEMORY;
    }
#if ATOMIC_IN_MEMIC
    a2a->net.atomic.mr =
        ibv_reg_dm_mr(ctx->shared_pd, a2a->net.atomic.counters, 0, size,
                      IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE |
                          IBV_ACCESS_ZERO_BASED);

#else
    a2a->net.atomic.mr =
        ibv_reg_mr(ctx->shared_pd, a2a->net.atomic.counters, size,
                   IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE);
#endif

    if (!a2a->net.atomic.mr) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "failed to register atomic couters array");
#if ATOMIC_IN_MEMIC
        ibv_free_dm(a2a->net.atomic.counters);
#else
        ucc_free(a2a->net.atomic.counters);
#endif
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static void ucc_tl_mlx5_dm_chunk_init(ucc_mpool_t *mp,        //NOLINT
                                      void *obj, void *chunk) //NOLINT
{
    ucc_tl_mlx5_dm_chunk_t *c    = (ucc_tl_mlx5_dm_chunk_t *)obj;
    ucc_tl_mlx5_team_t     *team =
        ucc_container_of(mp, ucc_tl_mlx5_team_t, dm_pool);

    c->addr = (uintptr_t)PTR_OFFSET(
        (UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host) ? team->dm_ptr : NULL,
        team->dm_offset);
    c->posted_sends    = 0;
    c->posted_all      = 0;
    c->completed_sends = 0;
    team->dm_offset    = PTR_OFFSET(
        team->dm_offset, UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_buf_size *
                             UCC_TL_MLX5_TEAM_LIB(team)->cfg.block_batch_size);
}

static ucc_mpool_ops_t ucc_tl_mlx5_dm_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_mlx5_dm_chunk_init,
    .obj_cleanup   = NULL};

void ucc_tl_mlx5_dm_pool_cleanup(ucc_tl_mlx5_team_t *team)
{
    if (!team->dm_ptr || !team->a2a) {
        return;
    }

    ucc_mpool_cleanup(&team->dm_pool, 1);

    ibv_dereg_mr(team->dm_mr);
    if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host) {
        ucc_free(team->dm_ptr);
    } else {
        ibv_free_dm(team->dm_ptr);
    }
    team->dm_ptr = NULL;
}

void ucc_tl_mlx5_dm_cleanup(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_dm_pool_cleanup(team);
    ucc_tl_mlx5_alltoall_atomic_free(team);
}

ucc_status_t ucc_tl_mlx5_dm_alloc_reg(struct ibv_context *ib_ctx,
                                      struct ibv_pd *pd, int dm_host,
                                      size_t buf_size, size_t *buf_num_p,
                                      struct ibv_dm **ptr, struct ibv_mr **mr,
                                      ucc_base_lib_t *lib)
{
    struct ibv_dm             *dm_ptr = NULL;
    struct ibv_mr             *dm_mr;
    struct ibv_device_attr_ex  attr;
    struct ibv_alloc_dm_attr   dm_attr;
    int                        max_chunks_to_alloc, min_chunks_to_alloc, i;

    if (dm_host) {
        max_chunks_to_alloc = (*buf_num_p == UCC_ULUNITS_AUTO)
                                  ? DM_HOST_AUTO_NUM_CHUNKS
                                  : *buf_num_p;
        dm_attr.length      = max_chunks_to_alloc * buf_size;
        dm_ptr              = ucc_malloc(dm_attr.length, "memic_host");
        if (!dm_ptr) {
            tl_debug(lib, " memic_host allocation failed");
            return UCC_ERR_NO_MEMORY;
        }

        dm_mr = ibv_reg_mr(pd, dm_ptr, dm_attr.length,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!dm_mr) {
            tl_debug(lib, "failed to reg host memory");
            ucc_free(dm_ptr);
            return UCC_ERR_NO_MESSAGE;
        }
        *buf_num_p = max_chunks_to_alloc;
    } else {
        attr.comp_mask = 0;
        if (ibv_query_device_ex(ib_ctx, NULL, &attr)) {
            tl_debug(lib, "failed to query device (errno=%d)", errno);
            return UCC_ERR_NO_MESSAGE;
        }
        if (!attr.max_dm_size) {
            tl_debug(lib, "device doesn't support dm allocation");
            return UCC_ERR_NO_RESOURCE;
        }
        max_chunks_to_alloc = min_chunks_to_alloc = *buf_num_p;
        if (*buf_num_p == UCC_ULUNITS_AUTO) {
            max_chunks_to_alloc =
                attr.max_dm_size / buf_size - 1; //keep reserved memory
            min_chunks_to_alloc = 1;
            if (!max_chunks_to_alloc) {
                tl_debug(lib,
                         "requested buffer size (=%ld) is too large, "
                         "should be set to be strictly less than %ld. "
                         "max allocation size is %ld",
                         buf_size, attr.max_dm_size / 2, attr.max_dm_size);
                return UCC_ERR_NO_RESOURCE;
            }
        }
        if (attr.max_dm_size < buf_size * min_chunks_to_alloc) {
            tl_debug(lib,
                     "cannot allocate %i buffer(s) of size %ld, "
                     "max allocation size is %ld",
                     min_chunks_to_alloc, buf_size, attr.max_dm_size);
            return UCC_ERR_NO_MEMORY;
        }
        memset(&dm_attr, 0, sizeof(dm_attr));
        for (i = max_chunks_to_alloc; i >= min_chunks_to_alloc; i--) {
            dm_attr.length = i * buf_size;
            errno          = 0;
            dm_ptr         = ibv_alloc_dm(ib_ctx, &dm_attr);
            if (dm_ptr) {
                break;
            }
        }
        if (!dm_ptr) {
            tl_debug(lib,
                     "dev mem allocation failed, requested %ld, attr.max %zd, "
                     "errno %d",
                     dm_attr.length, attr.max_dm_size, errno);
            return errno == ENOMEM || errno == ENOSPC ? UCC_ERR_NO_MEMORY
                                                      : UCC_ERR_NO_MESSAGE;
        }
        dm_mr = ibv_reg_dm_mr(pd, dm_ptr, 0, dm_attr.length,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_ZERO_BASED);
        if (!dm_mr) {
            tl_debug(lib, "failed to reg memic");
            ibv_free_dm(dm_ptr);
            return UCC_ERR_NO_MESSAGE;
        }
        *buf_num_p = i;
    }
    *ptr = dm_ptr;
    *mr  = dm_mr;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_dm_init(ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t    *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_lib_config_t *cfg = &UCC_TL_MLX5_TEAM_LIB(team)->cfg;
    ucc_status_t              status;

    status = ucc_tl_mlx5_alltoall_atomic_alloc(team);
    if (UCC_OK != status) {
        return status;
    }

    status = ucc_tl_mlx5_dm_alloc_reg(
        ctx->shared_ctx, ctx->shared_pd, cfg->dm_host,
        cfg->dm_buf_size * cfg->block_batch_size, &cfg->dm_buf_num,
        &team->dm_ptr, &team->dm_mr, UCC_TL_TEAM_LIB(team));
    if (status != UCC_OK) {
        goto err_dm_alloc;
    }
    team->dm_offset = 0;
    // TODO: fix/check the case dm_host=true
    ucc_assert(!cfg->dm_host);
    status = ucc_mpool_init(
        &team->dm_pool, 0, sizeof(ucc_tl_mlx5_dm_chunk_t), 0,
        UCC_CACHE_LINE_SIZE, 1, cfg->dm_buf_num, &ucc_tl_mlx5_dm_ops,
        ctx->super.super.ucc_context->thread_mode, "mlx5 dm pool");
    if (status != UCC_OK) {
        tl_debug(UCC_TL_TEAM_LIB(team), "failed to init dm pool");
        goto err_mpool_init;
    }
    return UCC_OK;

err_mpool_init:
    ibv_dereg_mr(team->dm_mr);
    if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host) {
        ucc_free(team->dm_ptr);
    } else {
        ibv_free_dm(team->dm_ptr);
    }
    team->dm_ptr = NULL;
err_dm_alloc:
    ucc_tl_mlx5_alltoall_atomic_free(team);
    return status;
}
