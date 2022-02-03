/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tl_mhba_ib.h"

ucc_status_t ucc_tl_mhba_create_ibv_ctx(char **               ib_devname,
                                        struct ibv_context **ctx, ucc_base_lib_t *lib)
{
    struct ibv_device **       dev_list = ibv_get_device_list(NULL);
    struct mlx5dv_context_attr attr     = {};
    struct ibv_device *        ib_dev;

    if (!(*ib_devname)) {
        /* If no device was specified by name, use by default the first
           available device. */
        ib_dev = *dev_list;
        if (!ib_dev) {
            tl_error(lib,"No IB devices found");
            return UCC_ERR_NO_MESSAGE;
        }
        *ib_devname = (char *)ibv_get_device_name(ib_dev);
    } else {
        int i;
        for (i = 0; dev_list[i]; ++i)
            if (!strcmp(ibv_get_device_name(dev_list[i]), *ib_devname))
                break;
        ib_dev = dev_list[i];
        if (!ib_dev) {
            tl_error(lib,"IB device %s not found", *ib_devname);
            return UCC_ERR_NO_MESSAGE;
        }
    }

    /* Need to open the device with `MLX5DV_CONTEXT_FLAGS_DEVX` flag, as it is
       needed for mlx5dv_create_mkey() (See man pages of mlx5dv_create_mkey()). */

    attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
    *ctx       = mlx5dv_open_device(ib_dev, &attr);
    return UCC_OK;
}

int ucc_tl_mhba_check_port_active(struct ibv_context *ctx, int port_num)
{
    struct ibv_port_attr port_attr;

    ibv_query_port(ctx, port_num, &port_attr);
    if (port_attr.state == IBV_PORT_ACTIVE &&
        port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        return 1;
    }
    return 0;
}

int ucc_tl_mhba_get_active_port(struct ibv_context *ctx)
{
    struct ibv_device_attr device_attr;
    int                    i;

    ibv_query_device(ctx, &device_attr);
    for (i = 1; i <= device_attr.phys_port_cnt; i++) {
        if (ucc_tl_mhba_check_port_active(ctx, i)) {
            return i;
        }
    }
    return -1;
}




ucc_status_t ucc_tl_mhba_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                    uint16_t lid, int port, ucc_base_lib_t *lib)
{
    int                ret;
    struct ibv_qp_attr qp_attr;

    tl_debug(lib, "modify QP to INIT");
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state        = IBV_QPS_INIT;
    qp_attr.pkey_index      = 0;
    qp_attr.port_num        = port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
        tl_error(lib, "QP RESET->INIT failed");
        return UCC_ERR_NO_MESSAGE;
    }

    tl_debug(lib, "modify QP to RTR");

    memset((void *)&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state              = IBV_QPS_RTR;
    qp_attr.path_mtu              = IBV_MTU_4096;
    qp_attr.dest_qp_num           = qp_num;
    qp_attr.rq_psn                = 0x123;
    qp_attr.min_rnr_timer         = 20;
    qp_attr.max_dest_rd_atomic    = 1;
    qp_attr.ah_attr.dlid          = lid;
    qp_attr.ah_attr.sl            = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num      = port;

    ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret != 0) {
        tl_error(lib, "QP INIT->RTR failed (error %d)", ret);
        return UCC_ERR_NO_MESSAGE;
    }

    // Modify QP to RTS
    tl_debug(lib, "modify QP to RTS");
    qp_attr.qp_state      = IBV_QPS_RTS;
    qp_attr.timeout       = 10;
    qp_attr.retry_cnt     = 7;
    qp_attr.rnr_retry     = 7;
    qp_attr.sq_psn        = 0x123;
    qp_attr.max_rd_atomic = 1;

    ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret != 0) {
        tl_error(lib, "QP RTR->RTS failed");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mhba_init_dc_qps_and_connect(ucc_tl_mhba_team_t *team,
                                                 uint32_t *          qpn,
                                                 uint8_t             port_num)
{
    ucc_tl_mhba_context_t *ctx = UCC_TL_MHBA_TEAM_CTX(team);
    int                        i;
    struct ibv_qp_init_attr_ex attr_ex;
    struct mlx5dv_qp_init_attr attr_dv;
    struct ibv_qp_attr         qp_attr_to_init;
    struct ibv_qp_attr         qp_attr_to_rtr;
    struct ibv_qp_attr         qp_attr_to_rts;

    memset(&attr_ex, 0, sizeof(attr_ex));
    memset(&attr_dv, 0, sizeof(attr_dv));
    memset(&qp_attr_to_init, 0, sizeof(qp_attr_to_init));
    memset(&qp_attr_to_rtr, 0, sizeof(qp_attr_to_rtr));
    memset(&qp_attr_to_rts, 0, sizeof(qp_attr_to_rts));

    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = team->net.cq;
    attr_ex.recv_cq = team->net.cq;
    attr_ex.pd      = ctx->shared_pd;
    /* Max number of send wrs per QP:
       max_number of blocks + 1 for atomic + 1 for barrier + 1 for transpose
       TODO: check for leftovers case ??
    */
    attr_ex.cap.max_send_wr =
        (SQUARED(team->node.sbgp->group_size / 2 + 1) + 3) * MAX_OUTSTANDING_OPS *
		ucc_div_round_up(team->net.net_size, team->num_dci_qps);
    attr_ex.cap.max_send_sge = 1;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                             IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                             IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD;
    attr_dv.comp_mask |=
        MLX5DV_QP_INIT_ATTR_MASK_DC | MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS |
        MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
    attr_dv.create_flags |= MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;

    attr_dv.send_ops_flags = MLX5DV_QP_EX_WITH_RAW_WQE;
    qp_attr_to_init.qp_state   = IBV_QPS_INIT;
    qp_attr_to_init.pkey_index = 0;
    qp_attr_to_init.port_num   = port_num;
    qp_attr_to_init.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    qp_attr_to_rtr.qp_state          = IBV_QPS_RTR;
    qp_attr_to_rtr.path_mtu          = IBV_MTU_4096;
    qp_attr_to_rtr.min_rnr_timer     = 20;
    qp_attr_to_rtr.ah_attr.port_num  = port_num;
    qp_attr_to_rtr.ah_attr.is_global = 0;

    qp_attr_to_rts.qp_state      = IBV_QPS_RTS;
    qp_attr_to_rts.timeout       = 10; //todo - what value?
    qp_attr_to_rts.retry_cnt     = 7;
    qp_attr_to_rts.rnr_retry     = 7;
    qp_attr_to_rts.sq_psn        = 0x123;
    qp_attr_to_rts.max_rd_atomic = 1;

    //create DCIs
    for (i = 0; i < team->num_dci_qps; i++) {
        team->net.dcis[i].dci_qp =
            mlx5dv_create_qp(ctx->shared_ctx, &attr_ex, &attr_dv);
        if (!team->net.dcis[i].dci_qp) {
            tl_error(UCC_TL_TEAM_LIB(team), "Couldn't create DCI QP");
            goto fail;
        }
        // Turn DCI ibv_qp to ibv_qpex and ibv_mqpex
        team->net.dcis[i].dc_qpex = ibv_qp_to_qp_ex(team->net.dcis[i].dci_qp);
        if (!team->net.dcis[i].dc_qpex) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "Failed turn ibv_qp to ibv_qp_ex, error: %d", errno);
            goto fail;
        }
        team->net.dcis[i].dc_mqpex =
            mlx5dv_qp_ex_from_ibv_qp_ex(team->net.dcis[i].dc_qpex);
        if (!team->net.dcis[i].dc_mqpex) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "Failed turn ibv_qp_ex to mlx5dv_qp_ex, error: %d", errno);
            goto fail;
        }

        if (ibv_modify_qp(team->net.dcis[i].dci_qp, &qp_attr_to_init,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT) !=
            0) {
            tl_error(UCC_TL_TEAM_LIB(team), "Failed to modify init qp");
            goto fail;
        }

        if (ibv_modify_qp(team->net.dcis[i].dci_qp, &qp_attr_to_rtr,
                          IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV) != 0) {
            tl_error(UCC_TL_TEAM_LIB(team), "Failed to modify qp to rtr");
            goto fail;
        }

        if (ibv_modify_qp(team->net.dcis[i].dci_qp, &qp_attr_to_rts,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                              IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) !=
            0) {
            tl_error(UCC_TL_TEAM_LIB(team), "Failed to modify qp to rts");
            goto fail;
        }
    }

    //create DCT
    memset(&attr_ex, 0, sizeof(struct ibv_qp_init_attr_ex));
    memset(&attr_dv, 0, sizeof(struct mlx5dv_qp_init_attr));

    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = team->net.cq;
    attr_ex.recv_cq = team->net.cq;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_PD;
    attr_ex.pd = ctx->shared_pd;
    struct ibv_srq_init_attr srq_attr;
    memset(&srq_attr, 0, sizeof(struct ibv_srq_init_attr));
    srq_attr.attr.max_wr  = 1;
    srq_attr.attr.max_sge = 1;
    // SRQ isn't really needed since we don't use SEND and RDMA WRITE with IMM, but needed because it's DCT
    team->net.srq = ibv_create_srq(ctx->shared_pd, &srq_attr);
    if (team->net.srq == NULL) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "Failed to create Shared Receive Queue (SRQ)");
        goto fail;
    }
    attr_ex.srq = team->net.srq;

    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC;
    attr_dv.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCT;
    attr_dv.dc_init_attr.dct_access_key = DC_KEY;

    team->net.dct_qp =
        mlx5dv_create_qp(ctx->shared_ctx, &attr_ex, &attr_dv);
    if (team->net.dct_qp == NULL) {
        tl_error(UCC_TL_TEAM_LIB(team),"Couldn't create DCT QP errno=%d", errno);
        goto srq_fail;
    }

    if (ibv_modify_qp(team->net.dct_qp, &qp_attr_to_init,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
        tl_error(UCC_TL_TEAM_LIB(team), "Failed to modify init qp");
        goto dct_fail;
    }

    if (ibv_modify_qp(team->net.dct_qp, &qp_attr_to_rtr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV |
                          IBV_QP_MIN_RNR_TIMER) != 0) {
        tl_error(UCC_TL_TEAM_LIB(team), "Failed to modify init qp");
        goto dct_fail;
    }

    *qpn= team->net.dct_qp->qp_num;
    return UCC_OK;

dct_fail:
    if (ibv_destroy_qp(team->net.dct_qp)) {
        tl_error(UCC_TL_TEAM_LIB(team), "Couldn't destroy QP");
    }
srq_fail:
    if (ibv_destroy_srq(team->net.srq)) {
        tl_error(UCC_TL_TEAM_LIB(team), "Couldn't destroy SRQ");
    }
fail:
    for (i = i - 1; i >= 0; i--) {
        if (ibv_destroy_qp(team->net.dcis[i].dci_qp)) {
            tl_error(UCC_TL_TEAM_LIB(team), "Couldn't destroy QP");
        }
    }
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_tl_mhba_create_rc_qps(ucc_tl_mhba_team_t *team,
                                       uint32_t *          qpns)
{
    ucc_tl_mhba_context_t *ctx = UCC_TL_MHBA_TEAM_CTX(team);
    struct ibv_qp_init_attr qp_init_attr;
    int                     i;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    //todo change in case of non-homogenous ppn
    qp_init_attr.send_cq = team->net.cq;
    qp_init_attr.recv_cq = team->net.cq;
    qp_init_attr.cap.max_send_wr =
        (SQUARED(team->node.sbgp->group_size / 2 + 1) + 3) *
        MAX_OUTSTANDING_OPS; // TODO switch back to fixed tx/rx
    qp_init_attr.cap.max_recv_wr     = 0;
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.cap.max_recv_sge    = 0;
    qp_init_attr.cap.max_inline_data = 0;
    qp_init_attr.qp_type             = IBV_QPT_RC;

    team->net.rc_qps = ucc_malloc(sizeof(struct ibv_qp *) * team->net.net_size);
    if (!team->net.rc_qps) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate asr qps array");
        goto fail_after_malloc;
    }
    for (i = 0; i < team->net.net_size; i++) {
        team->net.rc_qps[i] =
            ibv_create_qp(ctx->shared_pd, &qp_init_attr);
        if (!team->net.rc_qps[i]) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failed to create qp for dest %d, errno %d", i, errno);
            goto qp_creation_failure;
        }
        qpns[i] = team->net.rc_qps[i]->qp_num;
    }
    return UCC_OK;

qp_creation_failure:
    for (i = i - 1; i >= 0; i--) {
        if (ibv_destroy_qp(team->net.rc_qps[i])) {
            tl_error(UCC_TL_TEAM_LIB(team), "Couldn't destroy QP");
        }
    }
    ucc_free(team->net.rc_qps);
fail_after_malloc:
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_tl_mhba_create_ah(struct ibv_ah **ah_ptr, uint16_t lid,
                                   uint8_t port_num, ucc_tl_mhba_team_t *team)
{
    ucc_tl_mhba_context_t *ctx = UCC_TL_MHBA_TEAM_CTX(team);
    struct ibv_ah_attr ah_attr;
    memset(&ah_attr, 0, sizeof(struct ibv_ah_attr));

    ah_attr.dlid          = lid;
    ah_attr.port_num      = port_num;
    ah_attr.is_global     = 0;
    ah_attr.grh.hop_limit = 0;

    *ah_ptr = ibv_create_ah(ctx->shared_pd, &ah_attr);
    if (!(*ah_ptr)) {
        tl_error(UCC_TL_TEAM_LIB(team), "Failed to create ah");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}
