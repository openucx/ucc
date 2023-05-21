/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"

static ucs_status_t
rcache_reg_mr(void *context, ucs_rcache_t *rcache, //NOLINT: rcache is unused
              void *arg, ucc_rcache_region_t *rregion,
              uint16_t flags) //NOLINT: flags is unused
{
    ucc_tl_mlx5_context_t *ctx         = (ucc_tl_mlx5_context_t *)context;
    void *                 addr        = (void *)rregion->super.start;
    ucc_tl_mlx5_reg_t *    mlx5_reg    = ucc_tl_mlx5_get_rcache_reg_data(
                                                                      rregion);
    size_t                 length      = (size_t)(rregion->super.end
                                                       - rregion->super.start);
    int *                  change_flag = (int *)arg;

    mlx5_reg->region = rregion;
    *change_flag     = 1;
    mlx5_reg->mr     = ibv_reg_mr(ctx->shared_pd, addr, length,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mlx5_reg->mr) {
        tl_error(ctx->super.super.lib, "failed to register memory");
        return UCS_ERR_NO_MESSAGE;
    }
    return UCS_OK;
}

static void rcache_dereg_mr(void *        context, //NOLINT: context is unused
                            ucc_rcache_t *rcache,  //NOLINT: rcache is unused
                            ucc_rcache_region_t *rregion)
{
    ucc_tl_mlx5_reg_t *mlx5_reg = ucc_tl_mlx5_get_rcache_reg_data(rregion);

    ucc_assert(mlx5_reg->region == rregion);
    ibv_dereg_mr(mlx5_reg->mr);
    mlx5_reg->mr = NULL;
}

ucc_status_t tl_mlx5_rcache_create(ucc_tl_mlx5_context_t *ctx)
{
    static ucc_rcache_ops_t ucc_rcache_ops = {.mem_reg     = rcache_reg_mr,
                                              .mem_dereg   = rcache_dereg_mr,
                                              .dump_region = NULL};
    ucc_rcache_params_t     rcache_params;
    ucc_status_t            status;

    rcache_params.region_struct_size =
        sizeof(ucc_rcache_region_t) + sizeof(ucc_tl_mlx5_reg_t);
    rcache_params.alignment          = UCS_PGT_ADDR_ALIGN;
    rcache_params.max_alignment      = getpagesize();
    rcache_params.ucm_event_priority = 1000;
    rcache_params.context            = (void *)ctx;
    rcache_params.ops                = &ucc_rcache_ops;
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED
                                            | UCM_EVENT_MEM_TYPE_FREE;

    status = ucc_rcache_create(&rcache_params, "reg cache", &ctx->rcache);

    if (status != UCC_OK) {
        tl_error(ctx->super.super.lib, "Failed to create reg cache");
        return status;
    }
    return UCC_OK;
}
