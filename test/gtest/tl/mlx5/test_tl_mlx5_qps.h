/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5.h"

class test_tl_mlx5_qp : public test_tl_mlx5 {
  public:
    ucc_tl_mlx5_ib_qp_conf_t qp_conf;

    virtual void SetUp()
    {
        test_tl_mlx5::SetUp();
        CHECK_TEST_STATUS();

        qp_conf.qp_rnr_retry  = 7;
        qp_conf.qp_rnr_timer  = 20;
        qp_conf.qp_retry_cnt  = 7;
        qp_conf.qp_timeout    = 18;
        qp_conf.qp_max_atomic = 1;
    }
};

typedef ucc_status_t (*ucc_tl_mlx5_create_umr_qp_fn_t)(
    struct ibv_context *ctx, struct ibv_pd *pd, struct ibv_cq *cq, int ib_port,
    struct ibv_qp **qp, ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_qp_connect_fn_t)(
    struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, int port,
    ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_create_rc_qp_fn_t)(
    struct ibv_context *ctx, struct ibv_pd *pd, struct ibv_cq *cq, int tx_depth,
    ucc_tl_mlx5_qp_t *qp, uint32_t *qpn, ucc_base_lib_t *lib);

class test_tl_mlx5_rc_qp : public test_tl_mlx5_qp {
  public:
    ucc_tl_mlx5_qp_t               qp     = {};
    ucc_tl_mlx5_qp_t               umr_qp = {};
    uint32_t                       qpn;
    ucc_tl_mlx5_ib_qp_conf_t       umr_qp_conf;
    int                            tx_depth;
    ucc_tl_mlx5_create_rc_qp_fn_t  create_rc_qp;
    ucc_tl_mlx5_qp_connect_fn_t    qp_connect;
    ucc_tl_mlx5_create_umr_qp_fn_t create_rc_umr_qp;

    virtual void SetUp()
    {
        qp.qp       = NULL;
        umr_qp.qp   = NULL;
        tx_depth    = 4;

        test_tl_mlx5_qp::SetUp();
        CHECK_TEST_STATUS();

        umr_qp_conf = qp_conf;

        create_rc_qp = (ucc_tl_mlx5_create_rc_qp_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_create_rc_qp");
        ASSERT_EQ(nullptr, dlerror());

        qp_connect = (ucc_tl_mlx5_qp_connect_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_qp_connect");
        ASSERT_EQ(nullptr, dlerror());

        create_rc_umr_qp = (ucc_tl_mlx5_create_umr_qp_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_create_umr_qp");
        ASSERT_EQ(nullptr, dlerror());

    }

    ~test_tl_mlx5_rc_qp()
    {
        if (qp.qp) {
            ibv_destroy_qp(qp.qp);
        }
        if (umr_qp.qp) {
            ibv_destroy_qp(umr_qp.qp);
        }
    }

    void create_qp()
    {
        GTEST_ASSERT_EQ(create_rc_qp(ctx, pd, cq, tx_depth, &qp, &qpn, &lib),
                        UCC_OK);
    }

    void create_umr_qp()
    {
        GTEST_ASSERT_EQ(
            create_rc_umr_qp(ctx, pd, cq, port, &umr_qp.qp, &umr_qp_conf, &lib),
            UCC_OK);
    }

    void connect_qp_loopback()
    {
        GTEST_ASSERT_EQ(
            qp_connect(qp.qp, qpn, port_attr.lid, port, &qp_conf, &lib),
            UCC_OK);
    };
};

typedef ucc_status_t (*ucc_tl_mlx5_init_dct_fn_t)(
    struct ibv_pd *pd, struct ibv_context *ctx, struct ibv_cq *cq,
    struct ibv_srq *srq, uint8_t port_num, struct ibv_qp **dct_qp,
    uint32_t *qpn, ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_init_dci_fn_t)(
    ucc_tl_mlx5_dci_t *dci, struct ibv_pd *pd, struct ibv_context *ctx,
    struct ibv_cq *cq, uint8_t port_num, int tx_depth,
    ucc_tl_mlx5_ib_qp_conf_t *qp_conf, ucc_base_lib_t *lib);

typedef ucc_status_t (*ucc_tl_mlx5_create_ah_fn_t)(struct ibv_pd * pd,
                                                   uint16_t        lid,
                                                   uint8_t         port_num,
                                                   struct ibv_ah **ah_ptr,
                                                   ucc_base_lib_t *lib);

class test_tl_mlx5_dc : public test_tl_mlx5_qp {
  public:
    struct ibv_qp             *dct_qp = nullptr;
    struct ibv_ah             *ah     = nullptr;
    struct ibv_srq            *srq    = nullptr;
    ucc_tl_mlx5_dci_t          dci    = {};
    uint32_t                   dct_qpn;
    ucc_tl_mlx5_init_dct_fn_t  init_dct;
    ucc_tl_mlx5_init_dci_fn_t  init_dci;
    ucc_tl_mlx5_create_ah_fn_t create_ah;

    virtual void SetUp()
    {
        struct ibv_srq_init_attr srq_attr;

        test_tl_mlx5_qp::SetUp();
        CHECK_TEST_STATUS();

        init_dct = (ucc_tl_mlx5_init_dct_fn_t)dlsym(tl_mlx5_so_handle,
                                                    "ucc_tl_mlx5_init_dct");
        ASSERT_EQ(nullptr, dlerror());

        init_dci = (ucc_tl_mlx5_init_dci_fn_t)dlsym(tl_mlx5_so_handle,
                                                    "ucc_tl_mlx5_init_dci");
        ASSERT_EQ(nullptr, dlerror());

        create_ah = (ucc_tl_mlx5_create_ah_fn_t)dlsym(tl_mlx5_so_handle,
                                                      "ucc_tl_mlx5_create_ah");
        ASSERT_EQ(nullptr, dlerror());

        memset(&srq_attr, 0, sizeof(struct ibv_srq_init_attr));
        srq_attr.attr.max_wr  = 1;
        srq_attr.attr.max_sge = 1;

        srq = ibv_create_srq(pd, &srq_attr);
        EXPECT_NE(nullptr, srq);
        dct_qp = NULL;
    }

    ~test_tl_mlx5_dc()
    {
        if (ah) {
            ibv_destroy_ah(ah);
        }
        if (dct_qp) {
            ibv_destroy_qp(dct_qp);
        }
        if (dci.dci_qp) {
            ibv_destroy_qp(dci.dci_qp);
        }

        if (srq) {
            ibv_destroy_srq(srq);
        }
    }
};
