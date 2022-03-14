/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"
#include "utils/ucc_math.h"
#include "schedule/ucc_schedule.h"
#include "tl_mlx5_ib.h"
#include <limits.h>
#include "tl_mlx5_coll.h"
#include "tl_mlx5_pd.h"
#include "utils/arch/cpu.h"

UCC_CLASS_INIT_FUNC(ucc_tl_mlx5_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *        config)
{
    ucc_tl_mlx5_context_config_t *tl_mlx5_config =
        ucc_derived_of(config, ucc_tl_mlx5_context_config_t);
    int   port = -1, devname_len;
    char *ib_devname = NULL, *pos;
    ucc_status_t status;
    char  tmp[128];

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_mlx5_config->super,
                              params->context);
    memcpy(&self->cfg, tl_mlx5_config, sizeof(*tl_mlx5_config));


    status = ucc_mpool_init(
        &self->req_mp, 0,
        ucc_max(sizeof(ucc_tl_mlx5_task_t), sizeof(ucc_tl_mlx5_schedule_t)), 0,
        UCC_CACHE_LINE_SIZE, 8, UINT_MAX, NULL, params->thread_mode,
        "tl_mlx5_req_mp");
    if (UCC_OK != status) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_mlx5_req mpool");
        goto err_mpool;
    }


    if (self->cfg.devices.count > 0) {
        ib_devname       = self->cfg.devices.names[0];
        pos              = strstr(ib_devname, ":");
        devname_len      = (int)(pos - ib_devname);
        strncpy(tmp, ib_devname, devname_len);
        tmp[devname_len] = '\0';
        ib_devname       = tmp;
        port             = atoi(pos + 1);
    }
    if (UCC_OK != ucc_tl_mlx5_create_ibv_ctx(&ib_devname, &self->ib_ctx,
                                             self->super.super.lib)) {
        tl_error(self->super.super.lib, "failed to allocate ibv_context");
        goto release_mpool;
    }
    if (port == -1) {
        port = ucc_tl_mlx5_get_active_port(self->ib_ctx);
    }
    self->ib_port = port;
    if (-1 == port || !ucc_tl_mlx5_check_port_active(self->ib_ctx, port)) {
        tl_error(self->super.super.lib, "no active ports found on %s",
                 ib_devname);
        goto destroy_context;
    }
    tl_debug(self->super.super.lib, "using %s:%d", ib_devname, port);

    self->ib_pd = ibv_alloc_pd(self->ib_ctx); // TODO only ASR
    if (!self->ib_pd) {
        tl_error(self->super.super.lib, "failed to allocate ib_pd");
        goto destroy_context;
    }
    tl_info(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;

destroy_context:
    ibv_close_device(self->ib_ctx);
release_mpool:
	ucc_mpool_cleanup(&self->req_mp, 1);
err_mpool:
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_mlx5_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);

    ucc_rcache_destroy(self->rcache);
    ucc_mpool_cleanup(&self->req_mp, 1);
    ucc_tl_mlx5_remove_shared_ctx_pd(self);

    if (ibv_dealloc_pd(self->ib_pd)) {
        tl_error(self->super.super.lib, "failed to dealloc PD errno %d", errno);
    }
    ibv_close_device(self->ib_ctx);
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

ucc_status_t ucc_tl_mlx5_context_create_epilog(ucc_base_context_t *context)
{
    ucc_tl_mlx5_context_t *ctx = ucc_derived_of(context, ucc_tl_mlx5_context_t);
    ucc_context_t         *core_ctx = context->ucc_context;
    ucc_subset_t           s;
    ucc_status_t           status;
    ucc_topo_t            *topo;
    ucc_sbgp_t            *sbgp;
    ucc_tl_team_t         *steam;
    ucc_coll_task_t       *req;
    char sock_path[L_tmpnam];
    int  sock;

    if (!core_ctx->service_team) {
        /*todo can be implemented using global OOB */
        tl_error(context->lib, "not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }

    ucc_assert(core_ctx->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB);

    s.map.type = UCC_EP_MAP_FULL;
    s.map.ep_num = core_ctx->params.oob.n_oob_eps;
    s.myrank = core_ctx->rank;

    status = ucc_topo_init(s, core_ctx->topo, &topo);
    if (UCC_OK != status) {
        tl_error(context->lib, "failed to init mlx5 ctx topo");
        return status;
    }

    sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_NODE);
    if (sbgp->status != UCC_SBGP_ENABLED) {
        status = UCC_OK;
        goto out;
    }

    if (sbgp->group_rank == MLX5_ASR_RANK) {
        tmpnam(sock_path); //TODO switch to mkstemp
        status = ucc_tl_mlx5_asr_socket_init(ctx, sbgp->group_size,
                                             &sock, sock_path);
        if (UCC_OK != status) {
            sock_path[0]='\0';
            tl_error(context->lib, "failed to init asr socket");
        }
    }
    steam = core_ctx->service_team;

    s.map = sbgp->map;
    s.myrank = sbgp->group_rank;
    status = UCC_TL_TEAM_IFACE(steam)->scoll.bcast(&steam->super, sock_path, sizeof(sock_path), MLX5_ASR_RANK,
                                                   s, &req);

    if (UCC_OK != status) {
        tl_error(context->lib, "failed to start mlx5 ctx bcast");
        goto out;
    }

    while (UCC_INPROGRESS == (status = ucc_collective_test(&req->super))) {
        ucc_context_progress(core_ctx);
    }
    ucc_collective_finalize(&req->super);

    if (UCC_OK != status) {
        tl_error(context->lib, "failure during mlx5 ctx bcast");
        goto out;
    }
    if (strlen(sock_path) == 0) {
        goto out;
    }

    status  = ucc_tl_mlx5_share_ctx_pd(ctx, sock_path, sbgp->group_size,
                                       sbgp->group_rank == MLX5_ASR_RANK, sock);
    if (status != UCC_OK) {
        tl_error(context->lib, "failed to share ctx and pd");
        goto out;
    }

    status = tl_mlx5_create_rcache(ctx);
    if (UCC_OK != status) {
        tl_error(context->lib, "failed to create rcache");
        goto rcache_create_err;
    }

    ucc_topo_cleanup(topo);
    return UCC_OK;

rcache_create_err:

out:
    ucc_topo_cleanup(topo);
    return status;
}
