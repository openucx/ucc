/**
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "tl_mlx5.h"
#include "tl_mlx5_mcast_coll.h"
#include "coll_score/ucc_coll_score.h"
#include "tl_mlx5_mcast_helper.h"
#include "p2p/ucc_tl_mlx5_mcast_p2p.h"
#include "mcast/tl_mlx5_mcast_helper.h"
#include "mcast/tl_mlx5_mcast_service_coll.h"
#include "mcast/tl_mlx5_mcast_one_sided_reliability.h"

ucc_status_t ucc_tl_mlx5_mcast_team_init(ucc_base_context_t *base_context,
                                         ucc_tl_mlx5_mcast_team_t **mcast_team,
                                         ucc_tl_mlx5_mcast_context_t *ctx,
                                         const ucc_base_team_params_t *team_params,
                                         ucc_tl_mlx5_mcast_coll_comm_init_spec_t *mcast_conf)
{
    ucc_tl_mlx5_mcast_coll_comm_init_spec_t  comm_spec     = *mcast_conf;
    ucc_tl_mlx5_mcast_coll_context_t        *mcast_context = &(ctx->mcast_context);
    ucc_tl_mlx5_mcast_coll_comm_init_spec_t *conf_params   = &comm_spec;
    ucc_context_t                           *context       =  base_context->ucc_context;
    ucc_tl_mlx5_context_t                   *tl_ctx        = ucc_derived_of(base_context,
                                                                            ucc_tl_mlx5_context_t);
    ucc_status_t                             status;
    ucc_subset_t                             set;
    ucc_tl_mlx5_mcast_team_t                *new_mcast_team;
    ucc_tl_mlx5_mcast_oob_p2p_context_t     *oob_p2p_ctx;
    ucc_tl_mlx5_mcast_coll_comm_t           *comm;
    int                                      i;

    if (!ctx->mcast_ctx_ready) {
        tl_debug(base_context->lib,
                "mcast context not available, base_context = %p",
                 base_context );
        return UCC_ERR_NO_RESOURCE;
    }

    /* Check if actual mcast resources are available (they might be NULL in FORCE mode) */
    if (!mcast_context->ctx || !mcast_context->pd) {
        tl_mlx5_mcast_log(ctx->mcast_enabled, base_context->lib, UCC_LOG_LEVEL_WARN,
                              "mcast context resources not available, cannot create mcast team");
        return UCC_ERR_NO_RESOURCE;
    }

    new_mcast_team = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_team_t), "new_mcast_team");

    if (!new_mcast_team) {
        return UCC_ERR_NO_MEMORY;
    }

    new_mcast_team->mcast_context = ctx;

    /* init p2p interface */
    conf_params->p2p_iface.send_nb = ucc_tl_mlx5_mcast_p2p_send_nb;
    conf_params->p2p_iface.recv_nb = ucc_tl_mlx5_mcast_p2p_recv_nb;

    oob_p2p_ctx = ucc_malloc(sizeof(ucc_tl_mlx5_mcast_oob_p2p_context_t),
                             "oob_p2p_ctx");
    if (!oob_p2p_ctx) {
        ucc_free(new_mcast_team);
        return UCC_ERR_NO_MEMORY;
    }

    oob_p2p_ctx->base_ctx       = context;
    oob_p2p_ctx->base_team      = team_params->team;
    oob_p2p_ctx->my_team_rank   = team_params->rank;
    oob_p2p_ctx->lib            = mcast_context->lib;
    set.myrank                  = team_params->rank;
    set.map                     = team_params->map;
    oob_p2p_ctx->subset         = set;
    conf_params->oob            = oob_p2p_ctx;
    conf_params->sx_sge         = 1;
    conf_params->rx_sge         = 2;
    conf_params->scq_moderation = 64;

    comm = (ucc_tl_mlx5_mcast_coll_comm_t*)
            ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_coll_comm_t) +
                       sizeof(struct pp_packet*)*(conf_params->wsize-1),
                       "ucc_tl_mlx5_mcast_coll_comm_t");
    if (!comm) {
        ucc_free(oob_p2p_ctx);
        ucc_free(new_mcast_team);
        return UCC_ERR_NO_MEMORY;
    }

    /* Initialize HCA copy resources to NULL */
    comm->hca_copy_cq = NULL;
    comm->hca_copy_qp = NULL;

    ucc_list_head_init(&comm->bpool);
    ucc_list_head_init(&comm->pending_q);

    comm->service_coll.bcast_post     = ucc_tl_mlx5_mcast_service_bcast_post;
    comm->service_coll.allgather_post = ucc_tl_mlx5_mcast_service_allgather_post;
    comm->service_coll.allreduce_post = ucc_tl_mlx5_mcast_service_allreduce_post;
    comm->service_coll.barrier_post   = ucc_tl_mlx5_mcast_service_barrier_post;
    comm->service_coll.coll_test      = ucc_tl_mlx5_mcast_service_coll_test;

    memcpy(&comm->params, conf_params, sizeof(*conf_params));

    comm->one_sided.reliability_enabled = conf_params->one_sided_reliability_enable;
    comm->one_sided.reliability_scheme_msg_threshold
                                        = conf_params->reliability_scheme_msg_threshold;
    comm->one_sided.hca_copy_enabled    = conf_params->hca_copy_enabled;
    comm->max_eager                     = conf_params->max_eager;
    comm->truly_zero_copy_coll_min_msg  = conf_params->truly_zero_copy_coll_min_msg;
    comm->cuda_mem_enabled              = conf_params->cuda_mem_enabled;
    comm->comm_id                       = team_params->id;
    comm->ctx                           = mcast_context;
    comm->context                       = ctx;
    comm->mcast_group_count             = ucc_min(conf_params->mcast_group_count, MAX_GROUP_COUNT);

    /* only bcast or allgather is supported not both of them together */
    if (ctx->mcast_bcast_enabled) {
        comm->bcast_comm.mcast_prepost_bucket_size
                                            = conf_params->mcast_prepost_bucket_size;
        comm->bcast_comm.wsize              = conf_params->wsize;
        comm->bcast_comm.max_push_send      = conf_params->max_push_send;
        comm->bcast_comm.truly_zero_copy_bcast_enabled =
            conf_params->truly_zero_copy_bcast_enabled;
    } else {
        comm->allgather_comm.mcast_prepost_bucket_size
                                            = conf_params->mcast_prepost_bucket_size;
        comm->allgather_comm.truly_zero_copy_allgather_enabled
                                            = conf_params->truly_zero_copy_allgather_enabled;
        comm->allgather_comm.max_push_send  = conf_params->max_push_send;
    }

    if (comm->cuda_mem_enabled && !(tl_ctx->supported_mem_types & UCC_BIT(UCC_MEMORY_TYPE_CUDA))) {
        tl_warn(mcast_context->lib, "cuda-aware mcast not available as gpu direct is not ready");
        status = UCC_ERR_NO_RESOURCE;
        goto cleanup;
    }

    comm->mcast.rcq = ibv_create_cq(mcast_context->ctx, comm->params.rx_depth, NULL, NULL, 0);
    if (!comm->mcast.rcq) {
        tl_mlx5_mcast_log(mcast_context->params.mcast_enabled, mcast_context->lib, UCC_LOG_LEVEL_ERROR,
                          "could not create recv cq, rx_depth %d, errno %d",
                          comm->params.rx_depth, errno);
        status = UCC_ERR_NO_RESOURCE;
        goto cleanup;
    }

    comm->mcast.scq = ibv_create_cq(mcast_context->ctx, comm->params.sx_depth, NULL, NULL, 0);
    if (!comm->mcast.scq) {
        ibv_destroy_cq(comm->mcast.rcq);
        tl_mlx5_mcast_log(mcast_context->params.mcast_enabled, mcast_context->lib, UCC_LOG_LEVEL_ERROR,
                          "could not create send cq, sx_depth %d, errno %d",
                          comm->params.sx_depth, errno);
        status = UCC_ERR_NO_RESOURCE;
        goto cleanup;
    }

    comm->rank                  = team_params->rank;
    comm->commsize              = team_params->size;
    comm->max_per_packet        = mcast_context->mtu;
    comm->bcast_comm.last_acked = comm->bcast_comm.last_psn = 0;
    comm->bcast_comm.racks_n    = comm->bcast_comm.sacks_n  = 0;
    comm->bcast_comm.child_n    = comm->bcast_comm.parent_n = 0;
    comm->p2p_ctx               = conf_params->oob;

    memcpy(&comm->p2p, &conf_params->p2p_iface,
            sizeof(ucc_tl_mlx5_mcast_p2p_interface_t));

    comm->dummy_packet.psn = UINT32_MAX;

    for (i=0; i< comm->bcast_comm.wsize; i++) {
        comm->r_window[i] = &comm->dummy_packet;
    }

    comm->lib                  = base_context->lib;
    new_mcast_team->mcast_comm = comm;

    *mcast_team = new_mcast_team;
    tl_debug(base_context->lib, "posted tl mcast team : %p", new_mcast_team);

    return UCC_OK;

cleanup:
    ucc_free(comm);
    ucc_free(new_mcast_team);
    ucc_free(oob_p2p_ctx);
    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_coll_setup_comm_resources(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_status_t      status;
    size_t            page_size;
    int               buf_size, i, ret;
    ucc_memory_type_t supported_mem_type;

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
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR, "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    ret = posix_memalign((void**)&comm->call_rsgs, page_size, sizeof(struct ibv_sge) *
                         comm->params.rx_depth * 2);
    if (ret) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR, "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    comm->pending_recv = 0;
    comm->buf_n        = comm->params.rx_depth * 2;

    /* check the impact of creating recv posted buffers on GPU vs CPU */
    /* Note: For staging-based operations, we always use HOST memory for staging buffers
     * to use fast memcpy operations, then do one final cudaMemcpy to user buffer. */
    supported_mem_type = UCC_MEMORY_TYPE_HOST;

    comm->grh_buf = ucc_malloc(GRH_LENGTH * sizeof(char), "grh");
    if (ucc_unlikely(!comm->grh_buf)) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "failed to allocate grh memory");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    status = ucc_mc_memset(comm->grh_buf, 0, GRH_LENGTH, UCC_MEMORY_TYPE_HOST);
    if (status != UCC_OK) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "could not cuda memset");
        goto error;
    }

    comm->grh_mr = ibv_reg_mr(comm->ctx->pd, comm->grh_buf, GRH_LENGTH,
                              IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->grh_mr) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "could not register device memory for GRH, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    status = ucc_mc_alloc(&comm->pp_buf_header, buf_size * comm->buf_n, supported_mem_type);
    comm->pp_buf = comm->pp_buf_header->addr;
    if (ucc_unlikely(status != UCC_OK)) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "failed to allocate cuda memory");
        goto error;
    }

    status = ucc_mc_memset(comm->pp_buf, 0, buf_size * comm->buf_n, supported_mem_type);
    if (status != UCC_OK) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "could not memset");
        goto error;
    }

    comm->pp_mr = ibv_reg_mr(comm->ctx->pd, comm->pp_buf, buf_size * comm->buf_n,
                             IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!comm->pp_mr) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "could not register pp_buf device mr, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    ret = posix_memalign((void**) &comm->pp, page_size, sizeof(struct
                         pp_packet) * comm->buf_n);
    if (ret) {
        tl_mlx5_mcast_log(comm->context->mcast_enabled, comm->lib, UCC_LOG_LEVEL_ERROR,
                          "posix_memalign failed");
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < comm->buf_n; i++) {
        ucc_list_head_init(&comm->pp[i].super);

        comm->pp[i].buf     = (uintptr_t) comm->pp_buf + i * buf_size;
        comm->pp[i].context = 0;

        ucc_list_add_tail(&comm->bpool, &comm->pp[i].super);
    }

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

    memset(comm->bcast_comm.parents,  0, sizeof(comm->bcast_comm.parents));
    memset(comm->bcast_comm.children, 0, sizeof(comm->bcast_comm.children));

    comm->bcast_comm.nacks_counter                = 0;
    comm->bcast_comm.n_mcast_reliable             = 0;
    comm->bcast_comm.reliable_in_progress         = 0;
    comm->bcast_comm.recv_drop_packet_in_progress = 0;
    comm->tx                                      = 0;

    /* Mark transport ready on this rank */
    comm->mcast_transport_ready = 1;

    /* init one-sided reliability after transport ready */
    if (comm->one_sided.reliability_enabled) {
        status = ucc_tl_mlx5_mcast_one_sided_reliability_init(comm);
        if ((status != UCC_OK) && (status != UCC_ERR_NOT_SUPPORTED)) {
    return status;
        }
    }

    return UCC_OK;

error:
    /* Skip cleanup here: resources may be only partially initialized and
     * ucc_tl_mlx5_clean_mcast_comm expects fully initialized structures.
     * The higher-level fallback logic will dispose of @comm later if needed. */
    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_process_comm_event(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_mcast_coll_comm_t *comm    = tl_team->mcast->mcast_comm;
    ucc_status_t                   status  = UCC_OK;
    struct mcast_group            *group;
    int                            i;

    for (i = 0; i < comm->mcast_group_count; i++) {
        if (comm->mcast.groups[i].lid != 0) {
            continue;
        }

        status = ucc_tl_mlx5_mcast_join_mcast_get_event(comm->ctx, &comm->event);
        if (status != UCC_OK) {
            goto failed;
        }

        if (!comm->event) {
            tl_error(comm->lib, "received NULL event after successful get_event call");
            status = UCC_ERR_NO_MESSAGE;
            goto failed;
        }

        group = (struct mcast_group *)comm->event->param.ud.private_data;

        ucc_assert(group != NULL);
        tl_debug(comm->lib, "found mcast group %p info in the retreived mcast join event", group);

        group->lid  = comm->event->param.ud.ah_attr.dlid;
        group->mgid = comm->event->param.ud.ah_attr.grh.dgid;

        if (rdma_ack_cm_event(comm->event) < 0) {
            tl_error(comm->lib, "rdma_ack_cm_event failed");
            status               = UCC_ERR_NO_MESSAGE;
            comm->event          = NULL;
            goto failed;
        }
        comm->event = NULL;
    }

    tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_READY;
    return UCC_OK;

failed:
    if (status < 0) {
        tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_FAILED;
    }
    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_team_test(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team  = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_status_t                   status   = UCC_OK;
    struct sockaddr_in6            net_addr = {0,};
    ucc_tl_mlx5_mcast_join_info_t *data     = NULL;
    ucc_tl_mlx5_mcast_coll_comm_t *comm     = tl_team->mcast->mcast_comm;
    int                            i;

    if (comm->rank == 0) {
        switch(tl_team->mcast_state) {
            case TL_MLX5_TEAM_STATE_MCAST_INIT:
            {
                for (i = 0; i < comm->mcast_group_count; i++) {
                    net_addr.sin6_family   = AF_INET6;
                    net_addr.sin6_flowinfo = comm->comm_id + i + 1;
                    status = ucc_tl_mlx5_mcast_join_mcast_post(comm->ctx, &net_addr, &comm->mcast.groups[i], 1);
                    if (status < 0) {
                        tl_error(comm->lib, "rank 0 is unable to join mcast group %d error %d",
                                 i, status);
                        tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_FAILED;
                        return UCC_INPROGRESS;
                    }
                    comm->mcast.groups[i].mcast_addr = net_addr;
                }

                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_TEST;
                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_TEST:
            {
                status = ucc_tl_mlx5_mcast_process_comm_event(team);
                if (status < UCC_OK) {
                    tl_error(comm->lib, "mcast_process_comm_event failed");
                }

                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_READY:
            case TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_FAILED:
            {

                data = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_join_info_t),
                                  "ucc_tl_mlx5_mcast_join_info_t");
                if (!data) {
                    tl_error(comm->lib, "unable to allocate memory for group setup info");
                    return UCC_ERR_NO_MEMORY;
                }

                comm->group_setup_info = data;

                if (tl_team->mcast_state == TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_READY) {
                    /* rank 0 bcast the lid/gid of all the groups to other processes */
                    data->status = UCC_OK;
                    for (i = 0; i < comm->mcast_group_count; i++) {
                        data->dgid[i] = comm->mcast.groups[i].mgid;
                        data->dlid[i] = comm->mcast.groups[i].lid;
                    }
                } else {
                    /* rank 0 bcast the failed status to other processes so others do not hang */
                    status = ucc_tl_mlx5_leave_mcast_groups(comm->ctx, comm);
                    if (status) {
                        tl_error(comm->lib, "couldn't leave mcast group");
                    }
                    if (comm->group_setup_info) {
                        ucc_free(comm->group_setup_info);
                        comm->group_setup_info = NULL;
                    }
                    data->status = UCC_ERR_NO_RESOURCE;
                }

                status = comm->service_coll.bcast_post(comm->p2p_ctx, data, sizeof(ucc_tl_mlx5_mcast_join_info_t),
                                                       0, &comm->group_setup_info_req);
                if (UCC_OK != status) {
                    tl_error(comm->lib, "unable to post bcast for group setup info");
                    ucc_free(comm->group_setup_info);
                    comm->group_setup_info = NULL;
                    return status;
                }

                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_GRP_BCAST_POST;

                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_BCAST_POST:
            {
                /* rank 0 polls bcast request and wait for its completion */
                status = comm->service_coll.coll_test(comm->group_setup_info_req);
                if (UCC_OK != status) {
                    /* bcast is not completed yet */
                    if (status < 0) {
                        ucc_free(comm->group_setup_info);
                        comm->group_setup_info = NULL;
                    }
                    return status;
                }

                if (comm->group_setup_info && comm->group_setup_info->status != UCC_OK) {
                    /* rank 0 was not able to join a mcast group so all
                     * the ranks should return */
                    ucc_free(comm->group_setup_info);
                    comm->group_setup_info = NULL;
                    return UCC_ERR_NO_RESOURCE;
                }

                if (comm->group_setup_info) {
                    ucc_free(comm->group_setup_info);
                    comm->group_setup_info = NULL;
                }

                /* Try to setup per-rank multicast communication resources. If this
                 * fails on any rank we still need to participate in the global
                 * readiness allreduce so that OTHER ranks learn about the
                 * failure and gracefully fall back. Therefore, don’t return
                 * immediately on error – instead record the failure by keeping
                 * mcast_transport_ready at 0 and continue. On success the helper
                 * already sets mcast_transport_ready to 1. */
                status = ucc_tl_mlx5_mcast_coll_setup_comm_resources(comm);
                if (UCC_OK != status) {
                    tl_debug(comm->lib,
                             "mcast_coll_setup_comm_resources failed on rank %d, will fallback",
                             comm->rank);
                    /* Ensure we report failure in the upcoming MIN-allreduce. */
                    comm->mcast_transport_ready = 0;
                }

                /* post readiness allreduce */
                status = comm->service_coll.allreduce_post(comm->p2p_ctx,
                                                           &comm->mcast_transport_ready,
                                                           &comm->transport_ready_global,
                                                           1, UCC_DT_INT32, UCC_OP_MIN,
                                                           &comm->transport_ready_req);
                if (status != UCC_OK) {
                    return status;
                }
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_RELIAB_SYNC;
                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_RELIAB_SYNC:
            {
                status = comm->service_coll.coll_test(comm->transport_ready_req);
                if (UCC_OK != status) {
                    if (status < 0) {
                        return status;
                    }
                    return status;
                }
                /* request was finalized inside coll_test; reset pointer */
                comm->transport_ready_req = NULL;
                if (comm->transport_ready_global == 0) {
                    /* Some ranks failed to set up multicast transport: release any
                     * partial resources created locally so that PD can be
                     * deallocated cleanly during context teardown. */
                    ucc_tl_mlx5_clean_mcast_comm(comm);
                    return UCC_ERR_NO_RESOURCE;
                }
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_RELIABLITY;
                return UCC_INPROGRESS;
            }
            case TL_MLX5_TEAM_STATE_MCAST_RELIABLITY:
            {
                if (comm->one_sided.reliability_enabled) {
                status = ucc_tl_mlx5_mcast_one_sided_reliability_test(comm);
                if (UCC_OK != status) {
                        if (status < 0) {
                            tl_debug(comm->lib, "reliability test failed on rank %d", comm->rank);
                            return status;
                        } else {
                            /* Still in progress */
                    return status;
                        }
                    }
                    tl_debug(comm->lib, "reliability test succeeded on rank %d", comm->rank);
                } else {
                    tl_debug(comm->lib, "reliability disabled, skipping test on rank %d", comm->rank);
                }

                tl_debug(comm->lib, "initialized tl mcast team: %p", tl_team);
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_READY;
            }

            case TL_MLX5_TEAM_STATE_MCAST_READY:
            case TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE:
            {
                return UCC_OK;
            }

            default:
            {
                tl_error(comm->lib, "unknown state during mcast team: %p create", tl_team);
                return UCC_ERR_NO_RESOURCE;
            }
        }
    } else {
        /* none rank 0 team create states */
        switch(tl_team->mcast_state) {
            case TL_MLX5_TEAM_STATE_MCAST_INIT:
            {
                /* none 0 ranks bcast post to wait for rank 0 for lid/gid
                 * of the mcast group */
                data = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_join_info_t),
                                  "ucc_tl_mlx5_mcast_join_info_t");
                if (!data) {
                    tl_error(comm->lib, "unable to allocate memory for group setup info");
                    return UCC_ERR_NO_MEMORY;
                }

                status = comm->service_coll.bcast_post(comm->p2p_ctx, data,
                                                       sizeof(ucc_tl_mlx5_mcast_join_info_t), 0,
                                                       &comm->group_setup_info_req);
                if (UCC_OK != status) {
                    tl_error(comm->lib, "unable to post bcast for group setup info");
                    ucc_free(data);
                    return status;
                }

                comm->group_setup_info = data;
                tl_team->mcast_state   = TL_MLX5_TEAM_STATE_MCAST_GRP_BCAST_POST;

                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_BCAST_POST:
            {
                /* none rank 0 processes poll bcast request and wait for its completion */
                status = comm->service_coll.coll_test(comm->group_setup_info_req);
                if (UCC_OK != status) {
                    /* bcast is not completed yet */
                    if (status < 0) {
                        ucc_free(comm->group_setup_info);
                        comm->group_setup_info = NULL;
                    }
                    return status;
                }

                data   = comm->group_setup_info;
                if (!data) {
                    tl_error(comm->lib, "group_setup_info is NULL after successful coll_test");
                    return UCC_ERR_NO_RESOURCE;
                }

                status = data->status;
                if (UCC_OK != status) {
                    /* rank 0 was not able to join a mcast group so all
                     * the ranks should return */
                    ucc_free(data);
                    comm->group_setup_info = NULL;
                    return status;
                }

                /* now it is time for none rank 0 to call rdma_join_multicast()
                 * for all the mcast groups */
                for (i = 0; i < comm->mcast_group_count; i++) {
                    memcpy(&net_addr.sin6_addr, &(data->dgid[i]), sizeof(struct in6_addr));
                    net_addr.sin6_family = AF_INET6;
                    status = ucc_tl_mlx5_mcast_join_mcast_post(comm->ctx, &net_addr, &comm->mcast.groups[i], 0);
                    if (status < 0) {
                        tl_error(comm->lib, "none-root rank is unable to join mcast group error %d",
                                 status);
                        ucc_free(data);
                        comm->group_setup_info = NULL;
                        return status;
                    }
                    comm->mcast.groups[i].mcast_addr = net_addr;
                }

                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_TEST;
                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_TEST:
            {
                status = ucc_tl_mlx5_mcast_process_comm_event(team);
                if (status < UCC_OK) {
                    tl_error(comm->lib, "mcast_process_comm_event failed");
                }

                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_FAILED:
            {
                status = ucc_tl_mlx5_leave_mcast_groups(comm->ctx, comm);
                if (status) {
                    tl_error(comm->lib, "couldn't leave mcast group");
                }
                return UCC_ERR_NO_RESOURCE;
            }

            case TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_READY:
            {
                status = ucc_tl_mlx5_mcast_coll_setup_comm_resources(comm);
                if (UCC_OK != status) {
                    tl_debug(comm->lib,
                             "mcast_coll_setup_comm_resources failed on rank %d, will fallback",
                             comm->rank);
                    comm->mcast_transport_ready = 0;
                }

                status = comm->service_coll.allreduce_post(comm->p2p_ctx,
                                                           &comm->mcast_transport_ready,
                                                           &comm->transport_ready_global,
                                                           1, UCC_DT_INT32, UCC_OP_MIN,
                                                           &comm->transport_ready_req);
                if (status != UCC_OK) {
                    return status;
                }
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_RELIAB_SYNC;
                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_RELIAB_SYNC:
            {
                status = comm->service_coll.coll_test(comm->transport_ready_req);
                if (UCC_OK != status) {
                    if (status < 0) {
                        return status;
                    }
                    return status;
                }
                /* request finalized inside coll_test */
                comm->transport_ready_req = NULL;
                if (comm->transport_ready_global == 0) {
                    ucc_tl_mlx5_clean_mcast_comm(comm);
                    return UCC_ERR_NO_RESOURCE;
                }
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_RELIABLITY;
                return UCC_INPROGRESS;
            }

            case TL_MLX5_TEAM_STATE_MCAST_RELIABLITY:
            {
                if (comm->one_sided.reliability_enabled) {
                status = ucc_tl_mlx5_mcast_one_sided_reliability_test(comm);
                if (UCC_OK != status) {
                        if (status < 0) {
                            tl_debug(comm->lib, "reliability test failed on rank %d", comm->rank);
                            return status;
                        } else {
                            /* Still in progress */
                    return status;
                        }
                    }
                    tl_debug(comm->lib, "reliability test succeeded on rank %d", comm->rank);
                } else {
                    tl_debug(comm->lib, "reliability disabled, skipping test on rank %d", comm->rank);
                }

                tl_debug(comm->lib, "initialized tl mcast team: %p", tl_team);
                tl_team->mcast_state = TL_MLX5_TEAM_STATE_MCAST_READY;
            }

            case TL_MLX5_TEAM_STATE_MCAST_READY:
            case TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE:
            {
                return UCC_OK;
            }

            default:
            {
                tl_error(comm->lib, "unknown state during mcast team: %p create", tl_team);
                return UCC_ERR_NO_RESOURCE;
            }
        }
    }
}
