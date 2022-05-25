/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_ib.h"

ucc_status_t ucc_tl_mlx5_create_ibv_ctx(char **              ib_devname,
                                        struct ibv_context **ctx,
                                        ucc_base_lib_t *     lib)
{
    struct ibv_device **       dev_list = ibv_get_device_list(NULL);
    struct mlx5dv_context_attr attr     = {};
    ucc_status_t               status   = UCC_OK;
    struct ibv_device *        ib_dev;

    if (!(*ib_devname)) {
        /* If no device was specified by name, use by default the first
           available device. */
        ib_dev = *dev_list;
        if (!ib_dev) {
            tl_error(lib, "No IB devices found");
            status = UCC_ERR_NO_MESSAGE;
            goto err;
        }
        *ib_devname = (char *)ibv_get_device_name(ib_dev);
    } else {
        int i;
        for (i = 0; dev_list[i]; ++i)
            if (!strcmp(ibv_get_device_name(dev_list[i]), *ib_devname))
                break;
        ib_dev = dev_list[i];
        if (!ib_dev) {
            tl_error(lib, "IB device %s not found", *ib_devname);
            status = UCC_ERR_NO_MESSAGE;
            goto err;
        }
    }

    /* Need to open the device with `MLX5DV_CONTEXT_FLAGS_DEVX` flag, as it is
       needed for mlx5dv_create_mkey() (See man pages of mlx5dv_create_mkey()). */

    attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
    *ctx       = mlx5dv_open_device(ib_dev, &attr);
err:
    if (dev_list) {
        ibv_free_device_list(dev_list);
    }
    return status;
}

int ucc_tl_mlx5_check_port_active(struct ibv_context *ctx, int port_num)
{
    struct ibv_port_attr port_attr;

    ibv_query_port(ctx, port_num, &port_attr);
    if (port_attr.state == IBV_PORT_ACTIVE &&
        port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        return 1;
    }
    return 0;
}

int ucc_tl_mlx5_get_active_port(struct ibv_context *ctx)
{
    struct ibv_device_attr device_attr;
    int                    i;

    ibv_query_device(ctx, &device_attr);
    for (i = 1; i <= device_attr.phys_port_cnt; i++) {
        if (ucc_tl_mlx5_check_port_active(ctx, i)) {
            return i;
        }
    }
    return -1;
}

ucc_status_t ucc_tl_mlx5_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
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

ucc_status_t ucc_tl_mlx5_init_dct(struct ibv_pd *pd, struct ibv_context *ctx,
                                  struct ibv_cq *cq, struct ibv_srq *srq,
                                  uint8_t port_num, struct ibv_qp **dct_qp,
                                  uint32_t *qpn, ucc_base_lib_t *lib)
{
    struct ibv_qp_init_attr_ex attr_ex;
    struct mlx5dv_qp_init_attr attr_dv;
    struct ibv_qp_attr         qp_attr_to_init;
    struct ibv_qp_attr         qp_attr_to_rtr;
    struct ibv_qp *            qp;

    memset(&attr_ex, 0, sizeof(struct ibv_qp_init_attr_ex));
    memset(&attr_dv, 0, sizeof(struct mlx5dv_qp_init_attr));
    memset(&qp_attr_to_init, 0, sizeof(qp_attr_to_init));
    memset(&qp_attr_to_rtr, 0, sizeof(qp_attr_to_rtr));

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

    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = cq;
    attr_ex.recv_cq = cq;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_PD;
    attr_ex.pd  = pd;
    attr_ex.srq = srq;

    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC;
    attr_dv.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCT;
    attr_dv.dc_init_attr.dct_access_key = DC_KEY;

    qp = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (qp == NULL) {
        tl_error(lib, "Couldn't create DCT QP errno=%d", errno);
        return UCC_ERR_NO_MESSAGE;
    }

    if (ibv_modify_qp(qp, &qp_attr_to_init,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
        tl_error(lib, "Failed to modify init qp");
        goto fail;
    }

    if (ibv_modify_qp(qp, &qp_attr_to_rtr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV |
                          IBV_QP_MIN_RNR_TIMER) != 0) {
        tl_error(lib, "Failed to modify init qp");
        goto fail;
    }

    *dct_qp = qp;
    *qpn    = qp->qp_num;
    return UCC_OK;

fail:
    if (ibv_destroy_qp(qp)) {
        tl_error(lib, "Couldn't destroy QP");
    }
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_tl_mlx5_init_dci(ucc_tl_mlx5_dci_t *dci, struct ibv_pd *pd,
                                  struct ibv_context *ctx, struct ibv_cq *cq,
                                  uint8_t port_num, int tx_depth,
                                  ucc_base_lib_t *lib)
{
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

    attr_ex.qp_type          = IBV_QPT_DRIVER;
    attr_ex.send_cq          = cq;
    attr_ex.recv_cq          = cq;
    attr_ex.pd               = pd;
    attr_ex.cap.max_send_wr  = tx_depth;
    attr_ex.cap.max_send_sge = 1;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                             IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                             IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD;
    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC |
                         MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS |
                         MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
    attr_dv.create_flags |= MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;

    attr_dv.send_ops_flags     = MLX5DV_QP_EX_WITH_RAW_WQE;
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
    qp_attr_to_rts.timeout       = 10;
    qp_attr_to_rts.retry_cnt     = 7;
    qp_attr_to_rts.rnr_retry     = 7;
    qp_attr_to_rts.sq_psn        = 0x123;
    qp_attr_to_rts.max_rd_atomic = 1;

    dci->dci_qp = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (!dci->dci_qp) {
        tl_error(lib, "Couldn't create DCI QP");
        return UCC_ERR_NO_MESSAGE;
    }
    // Turn DCI ibv_qp to ibv_qpex and ibv_mqpex
    dci->dc_qpex = ibv_qp_to_qp_ex(dci->dci_qp);
    if (!dci->dc_qpex) {
        tl_error(lib, "Failed turn ibv_qp to ibv_qp_ex, error: %d", errno);
        goto fail;
    }
    dci->dc_mqpex = mlx5dv_qp_ex_from_ibv_qp_ex(dci->dc_qpex);
    if (!dci->dc_mqpex) {
        tl_error(lib, "Failed turn ibv_qp_ex to mlx5dv_qp_ex, error: %d",
                 errno);
        goto fail;
    }

    if (ibv_modify_qp(dci->dci_qp, &qp_attr_to_init,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT) != 0) {
        tl_error(lib, "Failed to modify init qp");
        goto fail;
    }

    if (ibv_modify_qp(dci->dci_qp, &qp_attr_to_rtr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV) != 0) {
        tl_error(lib, "Failed to modify qp to rtr");
        goto fail;
    }

    if (ibv_modify_qp(dci->dci_qp, &qp_attr_to_rts,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) !=
        0) {
        tl_error(lib, "Failed to modify qp to rts");
        goto fail;
    }

    return UCC_OK;

fail:
    if (ibv_destroy_qp(dci->dci_qp)) {
        tl_error(lib, "Couldn't destroy QP");
    }
    return UCC_ERR_NO_MESSAGE;
}

ucc_status_t ucc_tl_mlx5_create_rc_qp(struct ibv_context *ctx,
                                      struct ibv_pd *pd, struct ibv_cq *cq,
                                      int ib_port, int tx_depth,
                                      ucc_tl_mlx5_qp_t *qp, uint32_t *qpn,
                                      ucc_base_lib_t *lib)
{
    struct ibv_qp_init_attr_ex attr_ex;
    struct mlx5dv_qp_init_attr attr_dv;

    memset(&attr_ex, 0, sizeof(attr_ex));
    memset(&attr_dv, 0, sizeof(attr_dv));

    attr_ex.qp_type = IBV_QPT_RC;
    attr_ex.send_cq = cq;
    attr_ex.recv_cq = cq;
    attr_ex.pd      = pd;
    /* Max number of send wrs per QP:
       max_number of blocks * 2 (1 send + 1 transpose) + 1 for atomic + 1 for barrier
       TODO: check for leftovers case ??
    */
    attr_ex.cap.max_send_wr  = tx_depth;
    attr_ex.cap.max_send_sge = 1;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                             IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                             IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD;
    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS |
                         MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
    attr_dv.create_flags |= MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
    attr_dv.send_ops_flags = MLX5DV_QP_EX_WITH_RAW_WQE;

    qp->qp = mlx5dv_create_qp(ctx, &attr_ex, &attr_dv);
    if (!qp->qp) {
        tl_error(lib, "failed to create RC QP errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    qp->qp_ex = ibv_qp_to_qp_ex(qp->qp);
    *qpn      = qp->qp->qp_num;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_create_ah(struct ibv_ah **ah_ptr, uint16_t lid,
                                   uint8_t port_num, ucc_tl_mlx5_team_t *team)
{
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    struct ibv_ah_attr     ah_attr;
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

ucc_status_t ucc_tl_mlx5_create_umr_qp(struct ibv_context *ctx,
                                       struct ibv_pd *pd, struct ibv_cq *cq,
                                       int ib_port, struct ibv_qp **qp,
                                       ucc_base_lib_t *lib)
{
    struct ibv_qp_init_attr_ex umr_init_attr_ex;
    struct mlx5dv_qp_init_attr umr_mlx5dv_qp_attr;
    struct ibv_port_attr       port_attr;
    ucc_status_t               status = UCC_OK;
    struct ibv_qp_ex *         qp_ex;

    memset(&umr_mlx5dv_qp_attr, 0, sizeof(umr_mlx5dv_qp_attr));
    memset(&umr_init_attr_ex, 0, sizeof(umr_init_attr_ex));

    umr_mlx5dv_qp_attr.comp_mask =
        MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_mlx5dv_qp_attr.create_flags   = 0;
    umr_mlx5dv_qp_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MR_LIST |
                                        MLX5DV_QP_EX_WITH_MR_INTERLEAVED |
                                        MLX5DV_QP_EX_WITH_RAW_WQE;

    umr_init_attr_ex.send_cq          = cq;
    umr_init_attr_ex.recv_cq          = cq;
    umr_init_attr_ex.cap.max_send_wr  = 1;
    umr_init_attr_ex.cap.max_recv_wr  = 1;
    umr_init_attr_ex.cap.max_send_sge = 1;
    umr_init_attr_ex.cap.max_recv_sge = 1;
    // `max_inline_data` determines the WQE size that the QP will support.
    // The 'max_inline_data' should be modified only when the number of
    // arrays to interleave is greater than 3.
    //TODO query the devices what is max supported
    umr_init_attr_ex.cap.max_inline_data =
        828; // the max number possible, Sergey Gorenko's email
    umr_init_attr_ex.qp_type = IBV_QPT_RC;
    umr_init_attr_ex.comp_mask =
        IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_init_attr_ex.pd = pd;
    umr_init_attr_ex.send_ops_flags |= IBV_QP_EX_WITH_SEND;
    *qp = mlx5dv_create_qp(ctx, &umr_init_attr_ex, &umr_mlx5dv_qp_attr);
    if (*qp == NULL) {
        tl_error(lib, "failed to create UMR QP");
        return UCC_ERR_NO_MESSAGE;
    }
    qp_ex = ibv_qp_to_qp_ex(*qp);
    if (qp_ex == NULL) {
        tl_error(lib, "failed to create UMR qp_ex");
        status = UCC_ERR_NO_MESSAGE;
        goto failure;
    }

    // Turning on the IBV_SEND_SIGNALED option, will cause the reported work comletion to be with MLX5DV_WC_UMR opcode.
    // The option IBV_SEND_INLINE is required by the current API.
    qp_ex->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    if (ibv_query_port(ctx, ib_port, &port_attr)) {
        tl_error(lib, "failed to get port info (errno=%d)", errno);
        status = UCC_ERR_NO_MESSAGE;
        goto failure;
    }
    status =
        ucc_tl_mlx5_qp_connect(*qp, (*qp)->qp_num, port_attr.lid, ib_port, lib);
    if (status != UCC_OK) {
        goto failure;
    }

    tl_debug(lib, "created UMR QP, cap.max_inline_data = %d",
             umr_init_attr_ex.cap.max_inline_data);

    return UCC_OK;

failure:
    if (ibv_destroy_qp(*qp)) {
        tl_error(lib, "failed to destroy UMR QP (errno=%d)", errno);
    }
    return status;
}
