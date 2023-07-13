/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "tl_mlx5.h"
#include "tl_mlx5_mcast_coll.h"
#include "tl_mlx5_mcast_helper.h"
#include "p2p/ucc_tl_mlx5_mcast_p2p.h"
#include "coll_score/ucc_coll_score.h"

static ucc_tl_mlx5_mcast_coll_comm_t *ucc_tl_mlx5_mcast_setup_comm(int rank, int commsize,
                                                                   ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                                   ucc_tl_mlx5_mcast_coll_comm_init_spec_t *params,
                                                                   int comm_id)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm;
    int                            i;
    size_t                         page_size;
    int                            buf_size;

    comm = (ucc_tl_mlx5_mcast_coll_comm_t*) ucc_malloc(sizeof(ucc_tl_mlx5_mcast_coll_comm_t) +
            sizeof(struct pp_packet*)*(params->wsize-1), "ucc_tl_mlx5_mcast_coll_comm_t");

    if (!comm) {
        return NULL;
    }

    memset(comm, 0, sizeof(ucc_tl_mlx5_mcast_coll_comm_t));

    ucc_list_head_init(&comm->bpool);
    ucc_list_head_init(&comm->pending_q);

    if (params->sx_depth < 2 * params->scq_moderation) {
        params->sx_depth = 2 * params->scq_moderation;
    }

    if (params->rx_depth < 2 * params->sx_depth || params->rx_depth < 2 *
            params->wsize) {
        params->rx_depth =  MAX(2 * params->sx_depth, 2 * params->wsize);
    }

    memcpy(&comm->params, params, sizeof(*params));

    comm->wsize     = params->wsize;
    comm->max_eager = params->max_eager;
    comm->comm_id   = comm_id;
    comm->ctx       = ctx;
    comm->grh_buf   = (char *)ucc_malloc(GRH_LENGTH * sizeof(char), "grh_buf");
    if (!comm->grh_buf) {
        ucc_free(comm);
        return NULL;
    }
    memset(comm->grh_buf, 0, GRH_LENGTH);
    
    comm->grh_mr = ibv_reg_mr(ctx->pd, comm->grh_buf, GRH_LENGTH,
                              IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->grh_mr) {
        tl_error(ctx->lib, "Could not register memory for GRH, errno %d", errno);
        goto error;
    }

    comm->rcq = ibv_create_cq(ctx->ctx, comm->params.rx_depth, NULL, NULL, 0);
    if (!comm->rcq) {
        tl_error(ctx->lib, "Could not create recv cq, rx_depth %d, errno %d",
                  comm->params.rx_depth, errno);
        goto error;
    }

    comm->scq = ibv_create_cq(ctx->ctx, comm->params.sx_depth, NULL, NULL, 0);
    if (!comm->scq) {
        tl_error(ctx->lib, "Could not create send cq, sx_depth %d, errno %d",
                  comm->params.sx_depth, errno);
        goto error;
    }

    comm->rank           = rank;
    comm->commsize       = commsize;
    comm->max_per_packet = ctx->mtu - GRH_LENGTH;
    comm->last_acked     = comm->last_psn = 0;
    comm->racks_n        = comm->sacks_n  = 0;
    comm->child_n        = comm->parent_n = 0;
    comm->p2p_ctx        = params->oob;

    memcpy(&comm->p2p, &params->p2p_iface, sizeof(ucc_tl_mlx5_mcast_p2p_interface_t));

    comm->dummy_packet.psn = UINT32_MAX;

    for (i=0; i< comm->wsize; i++) {
        comm->r_window[i] = &comm->dummy_packet;
    }

    if (ucc_tl_setup_mcast(comm) != UCC_OK) {
        goto error;
    }

    if (ucc_tl_mlx5_mcast_init_qps(ctx, comm) != UCC_OK) {
        goto error;
    }

    if (ucc_tl_mlx5_mcast_setup_qps(ctx, comm) != UCC_OK) {
        goto error;
    }

    page_size = ucc_get_page_size();
    buf_size  = ctx->mtu;

    // Comm receiving buffers.
    posix_memalign((void**)&comm->call_rwr, page_size, sizeof(struct ibv_recv_wr) *
                   comm->params.rx_depth);
    posix_memalign((void**)&comm->call_rsgs, page_size, sizeof(struct ibv_sge) *
                   comm->params.rx_depth * 2);

    comm->pending_recv = 0;
    comm->buf_n        = comm->params.rx_depth * 2;

    posix_memalign((void**) &comm->pp_buf, page_size, buf_size * comm->buf_n);

    memset(comm->pp_buf, 0, buf_size * comm->buf_n);
    
    comm->pp_mr = ibv_reg_mr(ctx->pd, comm->pp_buf, buf_size * comm->buf_n,
                             IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->pp_mr) {
        tl_error(ctx->lib, "Could not register pp_buf mr, errno %d", errno);
        goto error;
    }

    posix_memalign((void**) &comm->pp, page_size, sizeof(struct pp_packet) * comm->buf_n);

    for (i = 0; i < comm->buf_n; i++) {
        ucc_list_head_init(&comm->pp[i].super);

        comm->pp[i].buf     = (uintptr_t) comm->pp_buf + i * buf_size;
        comm->pp[i].context = 0;
        
        ucc_list_add_tail(&comm->bpool, &comm->pp[i].super);
    }

    comm->mcast.swr.wr.ud.ah          = comm->mcast.ah;
    comm->mcast.swr.num_sge           = 1;
    comm->mcast.swr.sg_list           = & comm->mcast.ssg;
    comm->mcast.swr.opcode            = IBV_WR_SEND_WITH_IMM;
    comm->mcast.swr.wr.ud.remote_qpn  = MULTICAST_QPN;
    comm->mcast.swr.wr.ud.remote_qkey = DEF_QKEY;
    comm->mcast.swr.next              = NULL;

    for (i = 0; i < comm->params.rx_depth; i++) {
        comm->call_rwr[i].sg_list         = &comm->call_rsgs[2 * i];
        comm->call_rwr[i].num_sge         = 2;
        comm->call_rwr[i].wr_id           = MCAST_BCASTRECV_WR;
        comm->call_rsgs[2 * i].length     = GRH_LENGTH;
        comm->call_rsgs[2 * i].addr       = (uintptr_t)comm->grh_buf;
        comm->call_rsgs[2 * i].lkey       = comm->grh_mr->lkey;
        comm->call_rsgs[2 * i + 1].lkey   = comm->pp_mr->lkey;
        comm->call_rsgs[2 * i + 1].length = comm->max_per_packet;
    }

    if (UCC_OK != ucc_tl_mlx5_mcast_post_recv_buffers(comm)) {
        goto error;
    }

    ctx->params.barrier(comm->p2p_ctx);

    memset(comm->parents,  0, sizeof(comm->parents));
    memset(comm->children, 0, sizeof(comm->children));

    comm->nacks_counter        = 0;
    comm->tx                   = 0;
    comm->n_prep_reliable      = 0;
    comm->n_mcast_reliable     = 0;
    comm->reliable_in_progress = false;

    comm->recv_drop_packet_in_progress = false;

    return comm;

error:
    ucc_tl_clean_mcast_comm(comm);
    return NULL;
}

    
static ucc_status_t ucc_tl_mlx5_mcast_coll_comm_init(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                     ucc_tl_mlx5_mcast_coll_comm_init_spec_t *params,
                                                     int rank, int commsize,
                                                     ucc_tl_mlx5_mcast_coll_comm_t **_comm, int comm_id)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm;

    comm = ucc_tl_mlx5_mcast_setup_comm(rank, commsize, ctx, params, comm_id);
    if (comm == NULL) {
        return UCC_ERR_NO_RESOURCE;
    }

    *_comm = comm;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_team_init(ucc_base_context_t *base_context,
                                         ucc_tl_mlx5_mcast_team_t **mcast_team,
                                         ucc_tl_mlx5_mcast_context_t *ctx,
                                         const ucc_base_team_params_t *params,
                                         ucc_tl_mlx5_mcast_coll_comm_init_spec_t *mcast_conf)
{
    ucc_status_t                            status;
    ucc_subset_t                            set;
    ucc_tl_mlx5_mcast_oob_p2p_context_t    *oob_p2p_ctx;
    ucc_tl_mlx5_mcast_coll_context_t       *mcast_context = &(ctx->mcast_context);
    ucc_tl_mlx5_mcast_coll_comm_init_spec_t comm_spec     = *mcast_conf;
    ucc_context_t                          *tl_context    =  base_context->ucc_context;
    ucc_tl_mlx5_mcast_team_t               *new_mcast_team;

    new_mcast_team = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_team_t), "new_mcast_team");

    if (!new_mcast_team) {
        return UCC_ERR_NO_MEMORY;
    }

    if (NULL == mcast_context) {
        tl_debug(base_context->lib, "ERROR: mcast context not available, base_context = %p \n",
                base_context );
        ucc_free(new_mcast_team);
        return UCC_ERR_NO_RESOURCE;
    }

    new_mcast_team->mcast_context = ctx;

    /* init p2p interface */
    comm_spec.p2p_iface.send_nb  = ucc_tl_mlx5_mcast_p2p_send_nb;
    comm_spec.p2p_iface.recv_nb  = ucc_tl_mlx5_mcast_p2p_recv_nb;
    comm_spec.p2p_iface.progress = ucc_tl_mlx5_mcast_p2p_progress;

    oob_p2p_ctx = ucc_malloc(sizeof(ucc_tl_mlx5_mcast_oob_p2p_context_t), "oob_p2p_ctx");
    if (!oob_p2p_ctx) {
        ucc_free(new_mcast_team);
        return UCC_ERR_NO_MEMORY;
    }

    oob_p2p_ctx->base_ctx     = tl_context;
    oob_p2p_ctx->base_team    = params->team;
    oob_p2p_ctx->my_team_rank = params->rank;
     
    set.myrank                = params->rank;
    set.map                   = params->map;
    oob_p2p_ctx->subset       = set;
    comm_spec.oob             = oob_p2p_ctx;

    comm_spec.sx_sge          = 1;
    comm_spec.rx_sge          = 2;
    comm_spec.scq_moderation  = 64;

    status = ucc_tl_mlx5_mcast_coll_comm_init(mcast_context, &comm_spec, params->rank, params->size,
            &new_mcast_team->mcast_comm, params->id);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(base_context->lib, "mcast group create failed\n");
        goto cleanup;
    }

    new_mcast_team->mcast_comm->lib = base_context->lib;
    *mcast_team = new_mcast_team;

    tl_debug(base_context->lib, "initialized tl mcast team : %p", new_mcast_team);

    return UCC_OK;

cleanup:

    return status;
}

