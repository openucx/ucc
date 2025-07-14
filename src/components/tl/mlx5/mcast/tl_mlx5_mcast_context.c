/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <inttypes.h>
#include "tl_mlx5_mcast.h"
#include "utils/arch/cpu.h"
#include <ucs/sys/string.h>
#include "core/ucc_service_coll.h"
#include "tl_mlx5.h"
#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_mcast_rcache.h"

#define UCC_TL_MLX5_MCAST_MAX_MTU_COUNT 5
int mtu_lookup[UCC_TL_MLX5_MCAST_MAX_MTU_COUNT][2] = {
    {256,  IBV_MTU_256},
    {512,  IBV_MTU_512},
    {1024, IBV_MTU_1024},
    {2048, IBV_MTU_2048},
    {4096, IBV_MTU_4096}
};

ucc_status_t ucc_tl_mlx5_mcast_context_init(ucc_tl_mlx5_mcast_context_t    *context,
                                            ucc_tl_mlx5_mcast_ctx_params_t *mcast_ctx_conf)
{
    ucc_status_t            status        = UCC_OK;
    struct ibv_device     **device_list   = NULL;
    struct ibv_device      *dev           = NULL;
    char                   *devname       = NULL;
    int                     is_ipv4       = 0;
    struct sockaddr_in     *in_src_addr   = NULL;
    struct rdma_cm_event   *revent        = NULL;
    int                     active_mtu    = 4096;
    int                     max_mtu       = 4096;
    ucc_tl_mlx5_mcast_coll_context_t *ctx = NULL;
    char                   *ib_devname    = NULL;
    int                     devname_len   = 0, ib_port = 1;
    struct ibv_port_attr    port_attr;
    struct ibv_device_attr  device_attr;
    struct sockaddr_storage ip_oib_addr;
    struct sockaddr_storage dst_addr;
    int                     num_devices;
    char                    addrstr[128];
    ucc_tl_mlx5_context_t  *mlx5_ctx;
    ucc_base_lib_t         *lib;
    int                     i;
    int                     ib_valid;
    const char             *dst;
    char                    tmp[128], *pos, *end_pos;

    mlx5_ctx = ucc_container_of(context, ucc_tl_mlx5_context_t, mcast);
    lib      = mlx5_ctx->super.super.lib;

    context->mcast_enabled           = mcast_ctx_conf->mcast_enabled;
    context->mcast_bcast_enabled     = mcast_ctx_conf->mcast_bcast_enabled;
    context->mcast_allgather_enabled = mcast_ctx_conf->mcast_allgather_enabled;
    if (context->mcast_bcast_enabled && context->mcast_allgather_enabled) {
        /* only a single collective type is supported at a time */
        context->mcast_allgather_enabled = 0;
    }

    if (mcast_ctx_conf->mcast_enabled == UCC_NO) {
        tl_debug(lib, "Mcast is disabled by the user");
        return UCC_ERR_NO_RESOURCE;
    }

    ctx = &(context->mcast_context);
    memset(ctx, 0, sizeof(ucc_tl_mlx5_mcast_coll_context_t));
    memcpy(&ctx->params, mcast_ctx_conf, sizeof(ucc_tl_mlx5_mcast_ctx_params_t));

    ctx->lib = lib;

    /* TODO unify all the contexts under TL mlx5 */
    device_list = ibv_get_device_list(&num_devices);
    if (!device_list || !num_devices) {
        tl_debug(lib, "no ib devices available");
        status = UCC_ERR_NOT_SUPPORTED;
        goto error;
    }

    if (!strcmp(mcast_ctx_conf->ib_dev_name, "")) {
        dev          = device_list[0];
        devname      = (char *)ibv_get_device_name(dev);
        ctx->devname = ucc_malloc(strlen(devname)+3, "devname");
        if (!ctx->devname) {
            status = UCC_ERR_NO_MEMORY;
            goto error;
        }
        memset(ctx->devname, 0, strlen(devname)+3);
        memcpy(ctx->devname, devname, strlen(devname));
        strncat(ctx->devname, ":1", 3);
        ctx->user_provided_ib = 0;
        ctx->ib_port          = 1;
    } else {
        ib_valid = 0;
        /* user has provided the devname now make sure it is valid */
        /* check if port number is also included and extract devname from user str */
        ib_devname  = mcast_ctx_conf->ib_dev_name;
        pos         = strstr(ib_devname, ":");
        if (!pos) {
            devname_len = sizeof(tmp) - 1;
        } else {
            devname_len = (int)(pos - ib_devname);
            pos++;
            errno = 0;
            ib_port = (int)strtol(pos, &end_pos, 0);
            if (errno != 0 || pos == end_pos || strcmp(end_pos,"\0") || ib_port < 0
                    || ib_port > UINT8_MAX ) {
                tl_warn(lib, "wrong device's port number");
                return UCC_ERR_INVALID_PARAM;
            }
        }
        ctx->ib_port = ib_port;
        strncpy(tmp, ib_devname, devname_len);
        tmp[devname_len] = '\0';
        ib_devname       = tmp;

        for (i = 0; device_list[i]; ++i) {
            if (!strcmp(ibv_get_device_name(device_list[i]), ib_devname)) {
                ib_valid = 1;
                break;
            }
        }
        if (!ib_valid) {
            tl_warn(lib, "ib device %s not found", mcast_ctx_conf->ib_dev_name);
            status = UCC_ERR_NOT_FOUND;
            ibv_free_device_list(device_list);
            goto error;
        }
        ctx->devname          = mcast_ctx_conf->ib_dev_name;
        ctx->user_provided_ib = 1;
    }

    ibv_free_device_list(device_list);

    status = ucc_tl_mlx5_probe_ip_over_ib(ctx->devname, &ip_oib_addr);
    if (UCC_OK != status) {
        tl_debug(lib, "failed to get ipoib interface for devname %s", ctx->devname);
        if (!ctx->user_provided_ib) {
            ucc_free(ctx->devname);
        }
        goto error;
    }

    is_ipv4     = (ip_oib_addr.ss_family == AF_INET) ? 1 : 0;
    in_src_addr = (struct sockaddr_in*)&ip_oib_addr;

    dst = inet_ntop((is_ipv4) ? AF_INET : AF_INET6,
                    &in_src_addr->sin_addr, addrstr, sizeof(addrstr) - 1);
    if (NULL == dst) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN, "inet_ntop failed");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    tl_debug(ctx->lib, "devname %s, ipoib %s", ctx->devname, addrstr);

    ctx->channel = rdma_create_event_channel();
    if (!ctx->channel) {
        tl_debug(lib, "rdma_create_event_channel failed, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    memset(&dst_addr, 0, sizeof(struct sockaddr_storage));
    dst_addr.ss_family = is_ipv4 ? AF_INET : AF_INET6;
    if (rdma_create_id(ctx->channel, &ctx->id, NULL, RDMA_PS_UDP)) {
        tl_debug(lib, "failed to create rdma id, errno %d", errno);
        status = UCC_ERR_NOT_SUPPORTED;
        goto error;
    }

    if (0 != rdma_resolve_addr(ctx->id, (struct sockaddr *)&ip_oib_addr,
                               (struct sockaddr *) &dst_addr, 1000)) {
        tl_debug(lib, "failed to resolve rdma addr, errno %d", errno);
        status = UCC_ERR_NOT_SUPPORTED;
        goto error;
    }

    if (rdma_get_cm_event(ctx->channel, &revent) < 0) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN,
                          "failed to get cm event, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    } else if (revent->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN, "cm event is not resolved");
        if (rdma_ack_cm_event(revent) < 0) {
            tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN, "rdma_ack_cm_event failed");
        }
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    if (rdma_ack_cm_event(revent) < 0) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN, "rdma_ack_cm_event failed");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    ctx->ctx = ctx->id->verbs;
    ctx->pd  = ibv_alloc_pd(ctx->ctx);
    if (!ctx->pd) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN, "failed to allocate pd");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    /* Determine MTU */
    if (ibv_query_port(ctx->ctx, ctx->ib_port, &port_attr)) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN,
                          "couldn't query port in ctx create, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    for (i = 0; i < UCC_TL_MLX5_MCAST_MAX_MTU_COUNT; i++) {
        if (mtu_lookup[i][1] == port_attr.max_mtu) {
            max_mtu = mtu_lookup[i][0];
        }
        if (mtu_lookup[i][1] == port_attr.active_mtu) {
            active_mtu = mtu_lookup[i][0];
        }
    }

    ctx->mtu      = active_mtu;
    ctx->port_lid = port_attr.lid;

    tl_debug(ctx->lib, "port active MTU is %d and port max MTU is %d",
             active_mtu, max_mtu);

    if (port_attr.max_mtu < port_attr.active_mtu) {
        tl_debug(ctx->lib, "port active MTU (%d) is smaller than port max MTU (%d)",
                 active_mtu, max_mtu);
    }

    if (ibv_query_device(ctx->ctx, &device_attr)) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN,
                          "failed to query device in ctx create, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    tl_debug(ctx->lib, "MTU %d, MAX QP WR: %d, max sqr_wr: %d, max cq: %d, max cqe: %d",
             ctx->mtu, device_attr.max_qp_wr, device_attr.max_srq_wr,
             device_attr.max_cq, device_attr.max_cqe);

    ctx->max_qp_wr = device_attr.max_qp_wr;

    status = ucc_mpool_init(&ctx->compl_objects_mp, 0, sizeof(ucc_tl_mlx5_mcast_p2p_completion_obj_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_coll_task_mpool_ops,
                            UCC_THREAD_SINGLE,
                            "ucc_tl_mlx5_mcast_p2p_completion_obj_t");
    if (ucc_unlikely(UCC_OK != status)) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN,
                          "failed to initialize compl_objects_mp mpool");
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }

    status = ucc_mpool_init(&ctx->mcast_req_mp, 0, sizeof(ucc_tl_mlx5_mcast_coll_req_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_coll_task_mpool_ops,
                            UCC_THREAD_SINGLE,
                            "ucc_tl_mlx5_mcast_coll_req_t");
    if (ucc_unlikely(UCC_OK != status)) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN,
                          "failed to initialize mcast_req_mp mpool");
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }

    ctx->rcache = NULL;
    status = ucc_tl_mlx5_mcast_setup_rcache(ctx);
    if (UCC_OK != status) {
        tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN, "failed to setup rcache");
        goto error;
    }

    tl_debug(ctx->lib, "multicast context setup complete: ctx %p", ctx);

    return UCC_OK;

error:
    if (ctx->pd) {
        ibv_dealloc_pd(ctx->pd);
        ctx->pd = NULL;
    }
    if (ctx->id) {
        rdma_destroy_id(ctx->id);
        ctx->id = NULL;
    }
    if (ctx->channel) {
        rdma_destroy_event_channel(ctx->channel);
        ctx->channel = NULL;
    }

    /* Context initialization failed */
    tl_mlx5_mcast_log(context->mcast_enabled, lib, UCC_LOG_LEVEL_WARN,
                      "mcast context initialization failed (status: %d)", status);

    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t ucc_tl_mlx5_mcast_clean_ctx(ucc_tl_mlx5_mcast_coll_context_t *ctx)
{
    tl_debug(ctx->lib, "cleaning mcast ctx: %p", ctx);

    if (ctx->rcache) {
        ucc_rcache_destroy(ctx->rcache);
        ctx->rcache = NULL;
    }

    if (ctx->pd) {
        if (ibv_dealloc_pd(ctx->pd)) {
            tl_mlx5_mcast_log(ctx->params.mcast_enabled, ctx->lib, UCC_LOG_LEVEL_ERROR,
                              "ibv_dealloc_pd failed errno %d", errno);
            return UCC_ERR_NO_RESOURCE;
        }
        ctx->pd = NULL;
    }

    if (ctx->id && rdma_destroy_id(ctx->id)) {
        tl_mlx5_mcast_log(ctx->params.mcast_enabled, ctx->lib, UCC_LOG_LEVEL_ERROR,
                          "rdma_destroy_id failed errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    ctx->id = NULL;

    if (ctx->channel) {
        rdma_destroy_event_channel(ctx->channel);
        ctx->channel = NULL;
    }

   if (ctx->devname && !ctx->user_provided_ib) {
        ucc_free(ctx->devname);
        ctx->devname = NULL;
    }

    return UCC_OK;
}
