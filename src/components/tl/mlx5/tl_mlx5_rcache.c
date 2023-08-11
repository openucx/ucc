/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"

static ucs_status_t
rcache_reg_mr(void *context, ucc_rcache_t *rcache, //NOLINT: rcache is unused
              void *arg, ucc_rcache_region_t *rregion,
              uint16_t flags) //NOLINT: flags is unused
{
    ucc_tl_mlx5_context_t       *ctx          =
                                              (ucc_tl_mlx5_context_t *)context;
    void                        *addr         = (void *)rregion->super.start;
    size_t                       length       = (size_t)(rregion->super.end
                                                       - rregion->super.start);
    int                         *change_flag  = (int *)arg;
    ucc_tl_mlx5_rcache_region_t *mlx5_rregion = ucc_derived_of(rregion,
                                                  ucc_tl_mlx5_rcache_region_t);

    *change_flag         = 1;
    mlx5_rregion->reg.mr =
        ibv_reg_mr(ctx->shared_pd, addr, length,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mlx5_rregion->reg.mr) {
        tl_error(ctx->super.super.lib, "failed to register memory");
        return UCS_ERR_NO_MESSAGE;
    }
    return UCS_OK;
}

static void rcache_dereg_mr(void *context, //NOLINT: context is unused
                            ucc_rcache_t *rcache, //NOLINT: rcache is unused
                            ucc_rcache_region_t *rregion)
{
    ucc_tl_mlx5_rcache_region_t *mlx5_rregion =
        ucc_derived_of(rregion, ucc_tl_mlx5_rcache_region_t);

    ibv_dereg_mr(mlx5_rregion->reg.mr);
}

static void ucc_tl_mlx5_rcache_dump_region_cb(void *context, //NOLINT
                                              ucc_rcache_t *rcache, //NOLINT
                                              ucs_rcache_region_t *rregion,
                                              char *buf, size_t max)
{
    ucc_tl_mlx5_rcache_region_t *mlx5_rregion =
        ucc_derived_of(rregion, ucc_tl_mlx5_rcache_region_t);

    snprintf(buf, max, "bar ptr:%p", mlx5_rregion->reg.mr);
}

static ucc_rcache_ops_t ucc_rcache_ops = {
    .mem_reg     = rcache_reg_mr,
    .mem_dereg   = rcache_dereg_mr,
    .dump_region = ucc_tl_mlx5_rcache_dump_region_cb
};

ucc_status_t tl_mlx5_rcache_create(ucc_tl_mlx5_context_t *ctx)
{
    ucc_rcache_params_t     rcache_params;

    rcache_params.region_struct_size = sizeof(ucc_tl_mlx5_rcache_region_t);
    rcache_params.alignment          = UCS_PGT_ADDR_ALIGN;
    rcache_params.max_alignment      = ucc_get_page_size();
    rcache_params.ucm_event_priority = 1000;
    rcache_params.context            = (void *)ctx;
    rcache_params.ops                = &ucc_rcache_ops;
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED
                                            | UCM_EVENT_MEM_TYPE_FREE;

    return ucc_rcache_create(&rcache_params, "MLX5", &ctx->rcache);
}
