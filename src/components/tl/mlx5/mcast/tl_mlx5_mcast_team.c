/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "tl_mlx5.h"
#include "tl_mlx5_mcast_coll.h"
#include "coll_score/ucc_coll_score.h"
#include "tl_mlx5_mcast_helper.h"

ucc_status_t ucc_tl_mlx5_mcast_team_init(ucc_base_context_t           *base_context, /* NOLINT */
                                         ucc_tl_mlx5_mcast_team_t    **mcast_team, /* NOLINT */
                                         ucc_tl_mlx5_mcast_context_t  *ctx, /* NOLINT */
                                         const ucc_base_team_params_t *params, /* NOLINT */
                                         ucc_tl_mlx5_mcast_coll_comm_init_spec_t  *mcast_conf /* NOLINT */)
{
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_coll_setup_comm_resources(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_status_t status;
    size_t       page_size;
    int          buf_size, i, ret;

    status = ucc_tl_mlx5_mcast_init_qps(comm->ctx, comm);
    if (UCC_OK != status) {
        goto error;
    }

    status = ucc_tl_mlx5_mcast_setup_qps(comm->ctx, comm);
    if (UCC_OK != status) {
        goto error;
    }

    page_size = ucc_get_page_size();
    buf_size  = comm->ctx->mtu;

    // Comm receiving buffers.
    ret = posix_memalign((void**)&comm->call_rwr, page_size, sizeof(struct ibv_recv_wr) *
                         comm->params.rx_depth);
    if (ret) {
        tl_error(comm->ctx->lib, "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    ret = posix_memalign((void**)&comm->call_rsgs, page_size, sizeof(struct ibv_sge) *
                         comm->params.rx_depth * 2);
    if (ret) {
        tl_error(comm->ctx->lib, "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    comm->pending_recv = 0;
    comm->buf_n        = comm->params.rx_depth * 2;

    ret = posix_memalign((void**) &comm->pp_buf, page_size, buf_size * comm->buf_n);
    if (ret) {
        tl_error(comm->ctx->lib, "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    memset(comm->pp_buf, 0, buf_size * comm->buf_n);
    
    comm->pp_mr = ibv_reg_mr(comm->ctx->pd, comm->pp_buf, buf_size * comm->buf_n,
                             IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->pp_mr) {
        tl_error(comm->ctx->lib, "could not register pp_buf mr, errno %d", errno);
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }

    ret = posix_memalign((void**) &comm->pp, page_size, sizeof(struct
                         pp_packet) * comm->buf_n);
    if (ret) {
        tl_error(comm->ctx->lib, "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < comm->buf_n; i++) {
        ucc_list_head_init(&comm->pp[i].super);

        comm->pp[i].buf     = (uintptr_t) comm->pp_buf + i * buf_size;
        comm->pp[i].context = 0;
        
        ucc_list_add_tail(&comm->bpool, &comm->pp[i].super);
    }

    comm->mcast.swr.wr.ud.ah          = comm->mcast.ah;
    comm->mcast.swr.num_sge           = 1;
    comm->mcast.swr.sg_list           = &comm->mcast.ssg;
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

    status = ucc_tl_mlx5_mcast_post_recv_buffers(comm);
    if (UCC_OK != status) {
        goto error;
    }

    memset(comm->parents,  0, sizeof(comm->parents));
    memset(comm->children, 0, sizeof(comm->children));

    comm->nacks_counter                = 0;
    comm->tx                           = 0;
    comm->n_prep_reliable              = 0;
    comm->n_mcast_reliable             = 0;
    comm->reliable_in_progress         = 0;
    comm->recv_drop_packet_in_progress = 0;

    return status;

error:
    ucc_tl_mlx5_clean_mcast_comm(comm);
    return status;
}
