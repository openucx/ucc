/**
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define TL_MLX5_IB_PORT_INVALID -1

static ucc_status_t ucc_tl_mlx5_check_gpudirect_driver(ucc_base_lib_t *lib,
                                                       const char *file)
{
    ucc_status_t status = UCC_ERR_NO_RESOURCE;

    if (!access(file, F_OK)) {
        status = UCC_OK;
    }

    tl_debug(lib, "checking gpudirect driver: %s, status: %d %s", file,
             status, ucc_status_string(status));
    return status;
}

static ucc_status_t ucc_tl_mlx5_check_gpudirect_driver_cuda(ucc_base_lib_t *lib)
{
    /* Check peer memory driver is loaded, different driver versions use
     * different paths */
    if (UCC_OK == ucc_tl_mlx5_check_gpudirect_driver(lib,
                            "/sys/kernel/mm/memory_peers/nv_mem/version")) {
        return UCC_OK;
    } else if (UCC_OK == ucc_tl_mlx5_check_gpudirect_driver(lib,
                            "/sys/module/nvidia_peermem/version")) {
        return UCC_OK;
    } else if (UCC_OK == ucc_tl_mlx5_check_gpudirect_driver(lib,
                            "/sys/module/nv_peer_mem/version")) {
        return UCC_OK;
    }
    tl_debug(lib, "no gpudirect driver found, cuda memory is not supported");
    return UCC_ERR_NOT_SUPPORTED;
}

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
    self->sock                = 0;
    self->rcache              = NULL;
    self->shared_pd           = NULL;
    self->shared_ctx          = NULL;
    self->supported_mem_types = UCC_BIT(UCC_MEMORY_TYPE_HOST);

    if (UCC_OK == ucc_tl_mlx5_check_gpudirect_driver_cuda(self->super.super.lib)) {
        self->supported_mem_types |= UCC_BIT(UCC_MEMORY_TYPE_CUDA);
    }

    status = ucc_mpool_init(
        &self->req_mp, 0,
        ucc_max(sizeof(ucc_tl_mlx5_task_t), sizeof(ucc_tl_mlx5_schedule_t)), 0,
        UCC_CACHE_LINE_SIZE, 8, UINT_MAX, &ucc_coll_task_mpool_ops,
        params->thread_mode, "tl_mlx5_req_mp");
    if (UCC_OK != status) {
        tl_debug(self->super.super.lib,
                 "failed to initialize tl_mlx5_req mpool");
        return status;
    }

    if (self->cfg.enable_alltoall) {
        status = tl_mlx5_rcache_create(self);
        if (UCC_OK != status) {
            tl_debug(self->super.super.lib, "failed to create rcache");
            goto err_rcache;
        }
    } else {
        tl_debug(self->super.super.lib,
                 "alltoall is disabled by the env variable "
                 "`UCC_TL_MLX5_ALLTOALL_ENABLE`");
    }

    self->mcast.mcast_ctx_ready = 0;
    if (params->thread_mode == UCC_THREAD_SINGLE) {
        status = ucc_tl_mlx5_mcast_context_init(&(self->mcast), &(self->cfg.mcast_ctx_conf));
        if (UCC_OK != status) {
            tl_debug(self->super.super.lib, "failed to initialize mcast context");
        } else {
            self->mcast.mcast_ctx_ready = 1;
        }
    }
    return UCC_OK;

err_rcache:
    ucc_mpool_cleanup(&self->req_mp, 1);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_context_t)
{
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);
    if (self->rcache) {
        ucc_rcache_destroy(self->rcache);
    }

    if (UCC_OK != ucc_tl_mlx5_remove_shared_ctx_pd(self)) {
        tl_debug(self->super.super.lib, "failed to free ib ctx and pd");
    };

    if (self->sock) {
        close(self->sock);
    }

    ucc_mpool_cleanup(&self->req_mp, 1);

    if (self->mcast.mcast_ctx_ready) {
        ucc_tl_mlx5_mcast_clean_ctx(&self->mcast.mcast_context);
    }
}

UCC_CLASS_DEFINE(ucc_tl_mlx5_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_mlx5_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
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
            errno = 0;
            port = (int)strtol(pos, &end_pos, 10);
            if (errno != 0 || pos == end_pos) {
                tl_debug(ctx->super.super.lib, "wrong device's port number");
                return UCC_ERR_INVALID_PARAM;
            }
        }
        strncpy(tmp, ib_devname, devname_len);
        tmp[devname_len] = '\0';
        ib_devname       = tmp;
    }

    status = ucc_tl_mlx5_create_ibv_ctx(&ib_devname, &ctx->shared_ctx, ctx->super.super.lib);
    if (UCC_OK != status) {
        tl_debug(ctx->super.super.lib, "failed to allocate ibv_context");
        return status;
    }
    if (port == -1) {
        port = ucc_tl_mlx5_get_active_port(ctx->shared_ctx);
    }
    ctx->ib_port = port;
    if (-1 == port || !ucc_tl_mlx5_check_port_active(ctx->shared_ctx, port)) {
        tl_debug(ctx->super.super.lib, "no active ports found on %s",
                 ib_devname);
        goto destroy_context;
    }
    tl_debug(ctx->super.super.lib, "using %s:%d", ib_devname, port);

    ctx->shared_pd = ibv_alloc_pd(ctx->shared_ctx);
    if (!ctx->shared_pd) {
        tl_debug(ctx->super.super.lib, "failed to allocate ib_pd");
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

ucc_status_t ucc_tl_mlx5_context_ib_ctx_pd_setup(ucc_base_context_t *context)
{
    ucc_tl_mlx5_context_t *ctx = ucc_derived_of(context, ucc_tl_mlx5_context_t);
    ucc_context_t *  core_ctx           = context->ucc_context;
    const char *     template           = "/tmp/ucc.mlx5.XXXXXX";
    const char *     sockname           = "/sock";
    size_t           sock_dir_len       = strlen(template) + 1;
    size_t           sock_path_len      = sock_dir_len + strlen(sockname);
    size_t           sbcast_data_length = sizeof(int) + sock_path_len;
    char             sock_path[sock_path_len];
    ucc_subset_t     s;
    ucc_status_t     status;
    ucc_topo_t *     topo;
    ucc_sbgp_t *     sbgp;
    ucc_tl_team_t *  steam;
    ucc_coll_task_t *req;
    ucc_tl_mlx5_context_create_sbcast_data_t *sbcast_data;

    if (!ctx->cfg.enable_alltoall) {
        return UCC_OK;
    }

    if (!core_ctx->service_team) {
        tl_debug(context->lib, "failed to init ctx: need service team");
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_assert(core_ctx->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB);

    sbcast_data = (ucc_tl_mlx5_context_create_sbcast_data_t *)ucc_malloc(
        sbcast_data_length);
    if (!sbcast_data) {
        tl_debug(context->lib,
                 "failed to allocate buffer for sharing ib_ctx info");
        return UCC_ERR_NO_MEMORY;
    }

    memset(&s.map, 0, sizeof(ucc_ep_map_t));
    s.map.type   = UCC_EP_MAP_FULL;
    s.map.ep_num = core_ctx->params.oob.n_oob_eps;
    s.myrank     = core_ctx->rank;

    status = ucc_topo_init(s, core_ctx->topo, &topo);
    if (UCC_OK != status) {
        tl_debug(context->lib, "failed to init mlx5 ctx topo");
        goto err_topo;
    }

    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);

    ctx->shared_ctx  = NULL;
    ctx->shared_pd   = NULL;
    ctx->is_imported = sbgp->group_rank != PD_OWNER_RANK;

    if (!ctx->is_imported) {
        status = ucc_tl_mlx5_ib_ctx_pd_init(ctx);
        if (UCC_OK != status) {
            ctx->ib_port = TL_MLX5_IB_PORT_INVALID;
            goto start_bcast;
        }
        if (UCC_SBGP_NOT_EXISTS == sbgp->status) {
            goto topo_ppn_1;
        }
        ucc_strncpy_safe(sock_path, template, sock_dir_len);
        if (mkdtemp(sock_path) != NULL) {
            strncat(sock_path, sockname, sizeof(sock_path) - strlen(sock_path) - 1);
            status = ucc_tl_mlx5_socket_init(ctx, sbgp->group_size, &ctx->sock,
                                             sock_path);
            if (UCC_OK != status) {
                sock_path[0] = '\0';
                tl_debug(context->lib, "failed to init socket to share ib_ctx");
            }
        } else {
            tl_debug(context->lib, "failed to create tmp file for socket path");
            sock_path[0] = '\0';
        }
        memcpy(sbcast_data->sock_path, sock_path, sizeof(sock_path));
    }
start_bcast:
    sbcast_data->ib_port = ctx->ib_port;
    steam = core_ctx->service_team;
    s.map    = sbgp->map;
    s.myrank = sbgp->group_rank;

    status = UCC_TL_TEAM_IFACE(steam)->scoll.bcast(
        &steam->super, sbcast_data, sbcast_data_length, PD_OWNER_RANK, s, &req);

    if (UCC_OK != status) {
        tl_debug(context->lib, "failed to start mlx5 ctx bcast");
        goto err;
    }
    while (UCC_INPROGRESS == (status = ucc_collective_test(&req->super))) {
        ucc_context_progress(core_ctx);
    }
    ucc_collective_finalize_internal(req);

    if (UCC_OK != status) {
        tl_debug(context->lib, "failure during mlx5 ctx bcast");
        goto err;
    }

    ctx->ib_port = sbcast_data->ib_port;
    memcpy(sock_path, sbcast_data->sock_path, sizeof(sock_path));

    if (ctx->ib_port == TL_MLX5_IB_PORT_INVALID) {
        tl_debug(context->lib, "invalid ib port received");
        status = UCC_ERR_NO_RESOURCE;
        goto err_ib_ctx_pd_init;
    }

    if (strlen(sock_path) == 0) {
        tl_debug(context->lib, "failed to share ctx and pd");
        status = UCC_ERR_NO_RESOURCE;
        goto err;
    }
    status = ucc_tl_mlx5_share_ctx_pd(ctx, sock_path, sbgp->group_size,
                                      !ctx->is_imported, ctx->sock);

    if (UCC_OK != status) {
        goto err;
    }

    rmdir(sock_path);
topo_ppn_1:
    ucc_free(sbcast_data);
    ucc_topo_cleanup(topo);
    tl_debug(ctx->super.super.lib, "initialized tl context: %p", ctx);
    return UCC_OK;

err:
    ucc_tl_mlx5_remove_shared_ctx_pd(ctx);
    rmdir(sock_path);
    close(ctx->sock);
err_ib_ctx_pd_init:
    ucc_topo_cleanup(topo);
err_topo:
    ucc_free(sbcast_data);
    tl_debug(ctx->super.super.lib, "failed initialize tl context: %p", ctx);
    return status;
}

ucc_status_t ucc_tl_mlx5_context_create_epilog(ucc_base_context_t *context)
{
    return ucc_tl_mlx5_context_ib_ctx_pd_setup(context);
}
