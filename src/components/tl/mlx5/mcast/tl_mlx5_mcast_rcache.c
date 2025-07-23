/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_rcache.h"

static ucs_status_t ucc_tl_mlx5_mcast_coll_reg_mr(ucc_tl_mlx5_mcast_coll_context_t
                                                 *ctx, void *data, size_t data_size,
                                                  void **mr)
{
    *mr = ibv_reg_mr(ctx->pd, data, data_size, IBV_ACCESS_LOCAL_WRITE |
                     IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);

    tl_trace(ctx->lib, "external memory register: ptr %p, len %zd, mr %p",
             data,  data_size, (*mr));
    if (!*mr) {
        tl_error(ctx->lib, "failed to register MR");
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}


static ucc_status_t ucc_tl_mlx5_mcast_coll_dereg_mr(ucc_tl_mlx5_mcast_coll_context_t
                                                   *ctx, void *mr)
{
    if (ucc_unlikely(NULL == mr)) {
        tl_debug(ctx->lib, "external memory mr %p already deregistered", mr);
        return UCC_OK;
    }

    tl_trace(ctx->lib, "external memory deregister: mr %p", mr);

    if (ibv_dereg_mr(mr)) {
        tl_error(ctx->lib, "couldn't destroy mr %p", mr);
        return UCC_ERR_NO_RESOURCE;
    }

    return UCC_OK;
}

static ucs_status_t
ucc_tl_mlx5_mcast_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache, //NOLINT
                                    void *arg, ucs_rcache_region_t *rregion, //NOLINT
                                    uint16_t flags) //NOLINT
{
    ucc_tl_mlx5_mcast_rcache_region_t *region;
    void                              *address;
    size_t                             length;

    address = (void*)rregion->super.start;
    length  = (size_t)(rregion->super.end - rregion->super.start);
    region  = ucc_derived_of(rregion, ucc_tl_mlx5_mcast_rcache_region_t);

    return ucc_tl_mlx5_mcast_coll_reg_mr((ucc_tl_mlx5_mcast_coll_context_t *)context,
                                         address, length, &region->reg.mr);
}

static void ucc_tl_mlx5_mcast_rcache_mem_dereg_cb(void *context, ucc_rcache_t   //NOLINT
                                                 *rcache, ucc_rcache_region_t *rregion) //NOLINT
{
    ucc_tl_mlx5_mcast_rcache_region_t *region = ucc_derived_of(rregion,
                                                ucc_tl_mlx5_mcast_rcache_region_t);

    ucc_tl_mlx5_mcast_coll_dereg_mr((ucc_tl_mlx5_mcast_coll_context_t *)context,
                                     region->reg.mr);
}

static void ucc_tl_mlx5_mcast_rcache_dump_region_cb(void *context,  //NOLINT
                                                    ucc_rcache_t *rcache,   //NOLINT
                                                    ucc_rcache_region_t *rregion,   //NOLINT
                                                    char *buf,  //NOLINT
                                                    size_t max) //NOLINT
{
    ucc_tl_mlx5_mcast_rcache_region_t *region = ucc_derived_of(rregion,
                                           ucc_tl_mlx5_mcast_rcache_region_t);

    snprintf(buf, max, "bar ptr:%p", region->reg.mr);
}

ucc_status_t
ucc_tl_mlx5_mcast_mem_register(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                               void *addr, size_t length,
                               ucc_tl_mlx5_mcast_reg_t **reg)
{
    ucc_rcache_region_t               *rregion;
    ucc_tl_mlx5_mcast_rcache_region_t *region;
    ucc_status_t                       status;
    ucc_rcache_t                      *rcache;

    rcache = ctx->rcache;

    ucc_assert(rcache != NULL);

    status = ucc_rcache_get(rcache, (void *)addr, length, NULL, &rregion);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->lib, "ucc_rcache_get failed");
        return status;
    }

    region = ucc_derived_of(rregion, ucc_tl_mlx5_mcast_rcache_region_t);
    *reg   = &region->reg;

    tl_trace(ctx->lib, "memory register mr %p", (*reg)->mr);

    return UCC_OK;
}

void ucc_tl_mlx5_mcast_mem_deregister(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                      ucc_tl_mlx5_mcast_reg_t *reg)
{
    ucc_tl_mlx5_mcast_rcache_region_t *region;
    ucc_rcache_t                      *rcache;

    rcache = ctx->rcache;

    if (reg == NULL) {
        return;
    }

    ucc_assert(rcache != NULL);
    tl_trace(ctx->lib, "memory deregister mr %p", reg->mr);
    region = ucc_container_of(reg, ucc_tl_mlx5_mcast_rcache_region_t, reg);
    ucc_rcache_region_put(rcache, &region->super);
}

static ucc_rcache_ops_t ucc_tl_mlx5_rcache_ops = {
    .mem_reg     = ucc_tl_mlx5_mcast_rcache_mem_reg_cb,
    .mem_dereg   = ucc_tl_mlx5_mcast_rcache_mem_dereg_cb,
    .dump_region = ucc_tl_mlx5_mcast_rcache_dump_region_cb,
#ifdef UCS_HAVE_RCACHE_MERGE_CB
    .merge       = ucc_rcache_merge_cb_empty
#endif
};

ucc_status_t ucc_tl_mlx5_mcast_setup_rcache(ucc_tl_mlx5_mcast_coll_context_t *ctx)
{
    ucc_rcache_params_t rcache_params;

    ucc_rcache_set_default_params(&rcache_params);
    rcache_params.region_struct_size = sizeof(ucc_tl_mlx5_mcast_rcache_region_t);
    rcache_params.context            = ctx;
    rcache_params.ops                = &ucc_tl_mlx5_rcache_ops;
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED |
                                       UCM_EVENT_MEM_TYPE_FREE;

    return ucc_rcache_create(&rcache_params, "MLX5_MCAST", &ctx->rcache);
}
