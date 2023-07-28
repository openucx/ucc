/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "utils/ucc_math.h"
#include "schedule/ucc_schedule.h"
#include <limits.h>
#include "tl_mlx5_coll.h"
#include "utils/arch/cpu.h"
#include "tl_mlx5_pd.h"
#include "tl_mlx5_ib.h"

#define PD_OWNER_RANK 0

UCC_CLASS_INIT_FUNC(ucc_tl_mlx5_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *        config)
{
    ucc_tl_mlx5_context_config_t *tl_mlx5_config =
        ucc_derived_of(config, ucc_tl_mlx5_context_config_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_mlx5_config->super,
                              params->context);
    memcpy(&self->cfg, tl_mlx5_config, sizeof(*tl_mlx5_config));
    self->rcache     = NULL;
    self->shared_pd  = NULL;
    self->shared_ctx = NULL;

    status = ucc_mpool_init(
        &self->req_mp, 0,
        ucc_max(sizeof(ucc_tl_mlx5_task_t), sizeof(ucc_tl_mlx5_schedule_t)), 0,
        UCC_CACHE_LINE_SIZE, 8, UINT_MAX, &ucc_coll_task_mpool_ops,
        params->thread_mode, "tl_mlx5_req_mp");
    if (UCC_OK != status) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_mlx5_req mpool");
        return status;
    }

    status = ucc_tl_mlx5_mcast_context_init(&(self->mcast), &(self->cfg.mcast_ctx_conf));
    if (UCC_OK != status) {
        tl_error(self->super.super.lib,
                 "failed to initialize mcast context");
        return status;
    }

    tl_debug(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_context_t)
{
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);
    if (self->rcache) {
        ucc_rcache_destroy(self->rcache);
    }

    if (ucc_tl_mlx5_remove_shared_ctx_pd(self) != UCC_OK) {
        tl_error(self->super.super.lib, "failed to free ib ctx and pd");
    };

    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_mlx5_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_mlx5_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t *     attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }
    attr->topo_required = 1;
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_ib_ctx_pd_init(ucc_tl_mlx5_context_t *ctx)
{
    int          port       = -1;
    char *       ib_devname = NULL;
    int          devname_len;
    char         tmp[128], *pos, *end_pos;
    ucc_status_t status;

    if (ctx->cfg.devices.count > 0) {
        ib_devname  = ctx->cfg.devices.names[0];
        pos         = strstr(ib_devname, ":");
        end_pos     = ib_devname + strlen(ib_devname);
        if (!pos) {
            devname_len = sizeof(tmp) - 1;
        } else {
            devname_len = (int)(pos - ib_devname);
            pos++;
            port = (int)strtol(pos, &end_pos, 10);
            if (errno != 0 || pos == end_pos) {
                tl_error(ctx->super.super.lib, "wrong device's port number");
                return UCC_ERR_INVALID_PARAM;
            }
        }
        strncpy(tmp, ib_devname, devname_len);
        tmp[devname_len] = '\0';
        ib_devname       = tmp;
    }

    status = ucc_tl_mlx5_create_ibv_ctx(&ib_devname, &ctx->shared_ctx, ctx->super.super.lib);
    if (UCC_OK != status) {
        tl_error(ctx->super.super.lib, "failed to allocate ibv_context");
        return status;
    }
    if (port == -1) {
        port = ucc_tl_mlx5_get_active_port(ctx->shared_ctx);
    }
    ctx->ib_port = port;
    if (-1 == port || !ucc_tl_mlx5_check_port_active(ctx->shared_ctx, port)) {
        tl_error(ctx->super.super.lib, "no active ports found on %s",
                 ib_devname);
        goto destroy_context;
    }
    tl_debug(ctx->super.super.lib, "using %s:%d", ib_devname, port);

    ctx->shared_pd = ibv_alloc_pd(ctx->shared_ctx);
    if (!ctx->shared_pd) {
        tl_error(ctx->super.super.lib, "failed to allocate ib_pd");
        goto destroy_context;
    }

    return UCC_OK;

destroy_context:
    ibv_close_device(ctx->shared_ctx);

    return UCC_ERR_NO_RESOURCE;
}

typedef struct ucc_tl_mlx5_context_create_sbcast_data {
    int  ib_port;
    char sock_path[];
} ucc_tl_mlx5_context_create_sbcast_data_t;

ucc_status_t ucc_tl_mlx5_context_create_epilog(ucc_base_context_t *context)
{
    ucc_tl_mlx5_context_t *ctx = ucc_derived_of(context, ucc_tl_mlx5_context_t);
    ucc_context_t *  core_ctx           = context->ucc_context;
    const char *     template           = "/tmp/ucc.mlx5.XXXXXX";
    const char *     sockname           = "/sock";
    size_t           sock_dir_len       = strlen(template) + 1;
    size_t           sock_path_len      = sock_dir_len + strlen(sockname);
    int              sock               = 0;
    size_t           sbcast_data_length = sizeof(int) + sock_path_len;
    char             sock_path[sock_path_len];
    ucc_subset_t     s;
    ucc_status_t     status;
    ucc_topo_t *     topo;
    ucc_sbgp_t *     sbgp;
    ucc_tl_team_t *  steam;
    ucc_coll_task_t *req;
    ucc_tl_mlx5_context_create_sbcast_data_t *sbcast_data;

    if (!core_ctx->service_team) {
        tl_debug(context->lib, "failed to init ctx: need service team");
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_assert(core_ctx->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB);

    sbcast_data = (ucc_tl_mlx5_context_create_sbcast_data_t *)ucc_malloc(
        sbcast_data_length);
    if (!sbcast_data) {
        tl_error(context->lib,
                 "failed to allocate buffer for sharing ib_ctx info");
        return UCC_ERR_NO_MEMORY;
    }

    memset(&s.map, 0, sizeof(ucc_ep_map_t));
    s.map.type   = UCC_EP_MAP_FULL;
    s.map.ep_num = core_ctx->params.oob.n_oob_eps;
    s.myrank     = core_ctx->rank;

    status = ucc_topo_init(s, core_ctx->topo, &topo);
    if (UCC_OK != status) {
        tl_error(context->lib, "failed to init mlx5 ctx topo");
        goto err_topo;
    }

    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    if (sbgp->status != UCC_SBGP_ENABLED) {
        status = UCC_OK;
        goto err;
    }

    ctx->shared_ctx  = NULL;
    ctx->shared_pd   = NULL;
    ctx->is_imported = sbgp->group_rank != PD_OWNER_RANK;

    if (!ctx->is_imported) {
        ucc_strncpy_safe(sock_path, template, sock_dir_len);
        if (mkdtemp(sock_path) != NULL) {
            status = ucc_tl_mlx5_ib_ctx_pd_init(ctx);
            if (status != UCC_OK) {
                goto err;
            }

            strncat(sock_path, sockname, sizeof(sock_path) - strlen(sock_path) - 1);
            status = ucc_tl_mlx5_socket_init(ctx, sbgp->group_size, &sock,
                                             sock_path);
            if (UCC_OK != status) {
                sock_path[0] = '\0';
                tl_error(context->lib, "failed to init socket to share ib_ctx");
            }
        } else {
            tl_error(context->lib, "failed to create tmp file for socket path");
            sock_path[0] = '\0';
        }
        sbcast_data->ib_port = ctx->ib_port;
        memcpy(sbcast_data->sock_path, sock_path, sizeof(sock_path));
    }
    steam = core_ctx->service_team;

    s.map    = sbgp->map;
    s.myrank = sbgp->group_rank;
    status   = UCC_TL_TEAM_IFACE(steam)->scoll.bcast(
        &steam->super, sbcast_data, sbcast_data_length, PD_OWNER_RANK, s, &req);

    if (UCC_OK != status) {
        tl_error(context->lib, "failed to start mlx5 ctx bcast");
        goto err;
    }

    while (UCC_INPROGRESS == (status = ucc_collective_test(&req->super))) {
        ucc_context_progress(core_ctx);
    }
    ucc_collective_finalize(&req->super);

    if (UCC_OK != status) {
        tl_error(context->lib, "failure during mlx5 ctx bcast");
        goto err;
    }

    ctx->ib_port = sbcast_data->ib_port;
    memcpy(sock_path, sbcast_data->sock_path, sizeof(sock_path));

    if (strlen(sock_path) == 0) {
        status = UCC_ERR_NO_MESSAGE;
        goto err;
    }

    status = ucc_tl_mlx5_share_ctx_pd(ctx, sock_path, sbgp->group_size,
                                      !ctx->is_imported, sock);
    if (!ctx->is_imported) {
        sock_path[sock_dir_len - 1] = '\0';
        rmdir(sock_path);
    }
    if (status != UCC_OK) {
        tl_error(context->lib, "failed to share ctx and pd");
        goto err;
    }

    status = tl_mlx5_rcache_create(ctx);
    if (UCC_OK != status) {
        tl_error(context->lib, "failed to create rcache");
        goto err;
    }

    ucc_free(sbcast_data);
    ucc_topo_cleanup(topo);
    close(sock);
    return UCC_OK;

err:
    ucc_tl_mlx5_remove_shared_ctx_pd(ctx);
    ucc_topo_cleanup(topo);
    close(sock);
err_topo:
    ucc_free(sbcast_data);
    return status;
}
