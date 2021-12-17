/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mhba.h"

static ucs_status_t rcache_reg_mr(void *context, ucs_rcache_t *rcache,
                                  void *arg, ucc_rcache_region_t *rregion,
                                  uint16_t flags)
{
    ucc_tl_mhba_context_t *ctx = (ucc_tl_mhba_context_t *)context;
    void *              addr = (void *)rregion->super.start;
    size_t length = (size_t)(rregion->super.end - rregion->super.start);
    ucc_tl_mhba_reg_t *mhba_reg    = ucc_tl_mhba_get_rcache_reg_data(rregion);
    int *           change_flag = (int *)arg;

    mhba_reg->region = rregion;
    *change_flag     = 1;
    mhba_reg->mr =
        ibv_reg_mr(ctx->shared_pd, addr, length,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mhba_reg->mr) {
        tl_error(ctx->super.super.lib, "failed to register memory");
        return UCS_ERR_NO_MESSAGE;
    }
    return UCS_OK;
}

static void rcache_dereg_mr(void *context, ucc_rcache_t *rcache,
                            ucc_rcache_region_t *rregion)
{
    ucc_tl_mhba_reg_t *mhba_reg = ucc_tl_mhba_get_rcache_reg_data(rregion);

    ucc_assert(mhba_reg->region == rregion);
    ibv_dereg_mr(mhba_reg->mr);
    mhba_reg->mr = NULL;
}

ucc_status_t tl_mhba_create_rcache(ucc_tl_mhba_context_t *ctx)
{
    static ucc_rcache_ops_t rcache_ucc_ops = {
        .mem_reg     = rcache_reg_mr,
        .mem_dereg   = rcache_dereg_mr,
        .dump_region = NULL
    };
    ucc_rcache_params_t rcache_params;
    ucc_status_t status;

    rcache_params.region_struct_size =
        sizeof(ucc_rcache_region_t) + sizeof(ucc_tl_mhba_reg_t);
    rcache_params.alignment     = UCS_PGT_ADDR_ALIGN;
    rcache_params.max_alignment = getpagesize();
    rcache_params.ucm_events = UCM_EVENT_VM_UNMAPPED | UCM_EVENT_MEM_TYPE_FREE;
    rcache_params.ucm_event_priority = 1000;
    rcache_params.context            = (void *)ctx;
    rcache_params.ops                = &rcache_ucc_ops;

    status = ucc_rcache_create(&rcache_params, "reg cache", &ctx->rcache);

    if (status != UCC_OK) {
        tl_error(ctx->super.super.lib,"Failed to create reg cache");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}
