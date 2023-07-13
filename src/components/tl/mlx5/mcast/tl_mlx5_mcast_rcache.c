/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_rcache.h"

static ucc_status_t ucc_tl_mlx5_mcast_coll_reg_mr(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                  void *data, size_t data_size, void **mr)
{
    *mr = ibv_reg_mr(ctx->pd, data, data_size, IBV_ACCESS_LOCAL_WRITE);

    tl_debug(ctx->lib, "External memory register: ptr %p, len %zd, mr %p",
            data,  data_size, (*mr));
    if (!*mr) {
        tl_error(ctx->lib, "TL/MCAST: Failed to register MR\n");
        return UCC_ERR_NO_MEMORY;
    }
    
    return UCC_OK;
}


static ucc_status_t ucc_tl_mlx5_mcast_coll_dereg_mr(ucc_tl_mlx5_mcast_coll_context_t *ctx, void *mr)
{
    if(mr != NULL) {
        tl_debug(ctx->lib, "External memory deregister: mr %p", mr);
        ibv_dereg_mr(mr);
    } else {
        tl_debug(ctx->lib, "External memory mr %p already deregistered", mr);
    }

    return UCC_OK;
}

static ucs_status_t
ucc_tl_mlx5_mcast_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                    void *arg, ucs_rcache_region_t *rregion,
                                    uint16_t flags)
{
    ucc_tl_mlx5_mcast_rcache_region_t *region;
    void                              *address;
    size_t                             length;
    int                                ret;

    address = (void*)rregion->super.start;
    length  = (size_t)(rregion->super.end - rregion->super.start);
    region  = ucc_derived_of(rregion, ucc_tl_mlx5_mcast_rcache_region_t);

    ret     = ucc_tl_mlx5_mcast_coll_reg_mr((ucc_tl_mlx5_mcast_coll_context_t *)context, address,
                                            length, &region->reg.mr);
    if (ret < 0) {
        return UCS_ERR_INVALID_PARAM;
    } else {
        return UCS_OK;
    }
}

static void ucc_tl_mlx5_mcast_rcache_mem_dereg_cb(void *context, ucc_rcache_t *rcache,
                                                  ucc_rcache_region_t *rregion)
{
    ucc_tl_mlx5_mcast_rcache_region_t *region = ucc_derived_of(rregion,
                                                ucc_tl_mlx5_mcast_rcache_region_t);

    ucc_tl_mlx5_mcast_coll_dereg_mr((ucc_tl_mlx5_mcast_coll_context_t *)context, region->reg.mr);
}

static void ucc_tl_mlx5_mcast_rcache_dump_region_cb(void *context, ucc_rcache_t *rcache,
                                                    ucc_rcache_region_t *rregion, char *buf,
                                                    size_t max)
{
    ucc_tl_mlx5_mcast_rcache_region_t *region = ucc_derived_of(rregion,
                                           ucc_tl_mlx5_mcast_rcache_region_t);

    snprintf(buf, max, "bar ptr:%p", region->reg.mr);
}


ucc_status_t
ucc_tl_mlx5_mcast_mem_register(ucc_tl_mlx5_mcast_coll_context_t *ctx, void *addr,
                               size_t length, ucc_tl_mlx5_mcast_reg_t **reg)
{
    ucc_rcache_region_t               *rregion;
    ucc_tl_mlx5_mcast_rcache_region_t *region;
    ucc_status_t                       status;
    ucc_rcache_t                      *rcache;

    rcache = ctx->rcache;

    assert(rcache != NULL);

    status = ucc_rcache_get(rcache, (void *)addr, length, NULL,
                        &rregion);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->lib, "ucc_rcache_get failed\n");
        return UCC_ERR_INVALID_PARAM;
    }

    region = ucc_derived_of(rregion, ucc_tl_mlx5_mcast_rcache_region_t);
    *reg   = &region->reg;

    tl_debug(ctx->lib, "Memory register mr %p", (*reg)->mr);

    return UCC_OK;
}

ucc_status_t
ucc_tl_mlx5_mcast_mem_deregister(ucc_tl_mlx5_mcast_coll_context_t *ctx, ucc_tl_mlx5_mcast_reg_t *reg)
{
    ucc_tl_mlx5_mcast_rcache_region_t *region;
    ucc_rcache_t                      *rcache;

    rcache = ctx->rcache;

    if (reg == NULL) return UCC_OK;

    assert(rcache != NULL);

    tl_debug(ctx->lib, "Memory deregister mr %p", reg->mr);

    region = ucc_container_of(reg, ucc_tl_mlx5_mcast_rcache_region_t, reg);

    ucc_rcache_region_put(rcache, &region->super);

    return UCC_OK;
}

ucc_rcache_ops_t ucc_tl_mlx5_mcast_rcache_ops = {
    .mem_reg     = ucc_tl_mlx5_mcast_rcache_mem_reg_cb,
    .mem_dereg   = ucc_tl_mlx5_mcast_rcache_mem_dereg_cb,
    .dump_region = ucc_tl_mlx5_mcast_rcache_dump_region_cb
};

ucc_status_t ucc_tl_mlx5_mcast_setup_rcache(ucc_tl_mlx5_mcast_coll_context_t *ctx)
{
    ucc_status_t        status;
    ucc_rcache_t       *rcache;
    ucc_rcache_params_t rcache_params;

    rcache_params.alignment          = 64;
    rcache_params.ucm_event_priority = 1000;
    rcache_params.max_regions        = ULONG_MAX;
    rcache_params.max_size           = SIZE_MAX;
    rcache_params.region_struct_size = sizeof(ucc_tl_mlx5_mcast_rcache_region_t);
    rcache_params.max_alignment      = ucc_get_page_size();
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED | UCM_EVENT_MEM_TYPE_FREE;
    rcache_params.context            = ctx;
    rcache_params.ops                = &ucc_tl_mlx5_mcast_rcache_ops;
    rcache_params.flags              = 0;

    status = ucc_rcache_create(&rcache_params, "MCAST", &rcache);
    if (UCC_OK != status) {
        return status;
    }

    ctx->rcache = rcache;

    return UCC_OK;
}
