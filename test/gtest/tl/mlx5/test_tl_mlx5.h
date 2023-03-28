/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#ifndef TEST_TL_MLX5_H
#define TEST_TL_MLX5_H

#include "common/test_ucc.h"
#include "components/tl/mlx5/tl_mlx5.h"
#include "components/tl/mlx5/tl_mlx5_ib.h"

typedef ucc_status_t (*ucc_tl_mlx5_create_ibv_ctx_fn_t)(
    char **ib_devname, struct ibv_context **ctx, ucc_base_lib_t *lib);

typedef int (*ucc_tl_mlx5_get_active_port_fn_t)(struct ibv_context *ctx);

typedef ucc_status_t (*ucc_tl_mlx5_create_rc_qp_fn_t)(
    struct ibv_context *ctx, struct ibv_pd *pd, struct ibv_cq *cq, int tx_depth,
    ucc_tl_mlx5_qp_t *qp, uint32_t *qpn, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_qp_connect_fn_t)(
    struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, int port,
    ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_init_dct_fn_t)(
    struct ibv_pd *pd, struct ibv_context *ctx, struct ibv_cq *cq,
    struct ibv_srq *srq, uint8_t port_num, struct ibv_qp **dct_qp,
    uint32_t *qpn, ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_init_dci_fn_t)(
    ucc_tl_mlx5_dci_t *dci, struct ibv_pd *pd, struct ibv_context *ctx,
    struct ibv_cq *cq, uint8_t port_num, int tx_depth,
    ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_create_ah_fn_t)(struct ibv_pd  *pd,
                                                   uint16_t        lid,
                                                   uint8_t         port_num,
                                                   struct ibv_ah **ah_ptr,
                                                   ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_create_umr_qp_fn_t)(
    struct ibv_context *ctx, struct ibv_pd *pd, struct ibv_cq *cq, int ib_port,
    struct ibv_qp **qp, ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

class test_tl_mlx5 : public ucc::test {
    void *tl_mlx5_so_handle;
  public:
    ucc_base_lib_t                   lib;
    ucc_tl_mlx5_create_ibv_ctx_fn_t  create_ibv_ctx;
    ucc_tl_mlx5_get_active_port_fn_t get_active_port;
    ucc_tl_mlx5_create_rc_qp_fn_t    create_rc_qp;
    ucc_tl_mlx5_qp_connect_fn_t      qp_connect;
    ucc_tl_mlx5_init_dct_fn_t        init_dct;
    ucc_tl_mlx5_init_dci_fn_t        init_dci;
    ucc_tl_mlx5_create_ah_fn_t       create_ah;
    ucc_tl_mlx5_create_umr_qp_fn_t   create_umr_qp;
    ucc_tl_mlx5_ib_qp_conf_t         qp_conf;
    struct ibv_port_attr             port_attr;
    struct ibv_context *             ctx;
    struct ibv_pd *                  pd;
    struct ibv_cq *                  cq;
    int                              port;
    test_tl_mlx5();
    virtual ~test_tl_mlx5();
    virtual void SetUp() override;
};

#endif
