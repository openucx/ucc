/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5.h"
#include <dlfcn.h>

test_tl_mlx5::test_tl_mlx5()
{
    tl_mlx5_so_handle = NULL;
    ctx               = NULL;
    pd                = NULL;
    cq                = NULL;
}

void test_tl_mlx5::SetUp()
{
    char *       devname = NULL;
    ucc_status_t status;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ucc_strncpy_safe(lib.log_component.name, "GTEST_MLX5",
                     sizeof(lib.log_component.name));
    lib.log_component.log_level = UCC_LOG_LEVEL_ERROR;

    std::string path =
        std::string(ucc_global_config.component_path) + "/libucc_tl_mlx5.so";
    tl_mlx5_so_handle = dlopen(path.c_str(), RTLD_NOW);
    if (!tl_mlx5_so_handle) {
        GTEST_SKIP();
    }

    create_ibv_ctx = (ucc_tl_mlx5_create_ibv_ctx_fn_t)dlsym(
        tl_mlx5_so_handle, "ucc_tl_mlx5_create_ibv_ctx");
    ASSERT_EQ(nullptr, dlerror());

    get_active_port = (ucc_tl_mlx5_get_active_port_fn_t)dlsym(
        tl_mlx5_so_handle, "ucc_tl_mlx5_get_active_port");
    ASSERT_EQ(nullptr, dlerror());

    create_rc_qp = (ucc_tl_mlx5_create_rc_qp_fn_t)dlsym(
        tl_mlx5_so_handle, "ucc_tl_mlx5_create_rc_qp");
    ASSERT_EQ(nullptr, dlerror());

    qp_connect = (ucc_tl_mlx5_qp_connect_fn_t)dlsym(tl_mlx5_so_handle,
                                                    "ucc_tl_mlx5_qp_connect");
    ASSERT_EQ(nullptr, dlerror());

    init_dct = (ucc_tl_mlx5_init_dct_fn_t)dlsym(tl_mlx5_so_handle,
                                                "ucc_tl_mlx5_init_dct");
    ASSERT_EQ(nullptr, dlerror());

    init_dci = (ucc_tl_mlx5_init_dci_fn_t)dlsym(tl_mlx5_so_handle,
                                                "ucc_tl_mlx5_init_dci");
    ASSERT_EQ(nullptr, dlerror());

    create_ah = (ucc_tl_mlx5_create_ah_fn_t)dlsym(tl_mlx5_so_handle,
                                                  "ucc_tl_mlx5_create_ah");
    ASSERT_EQ(nullptr, dlerror());

    status = create_ibv_ctx(&devname, &ctx, &lib);
    if (UCC_OK != status) {
        std::cerr << "no ib devices";
        GTEST_SKIP();
    }
    port = get_active_port(ctx);
    ASSERT_GE(port, 0);

    ibv_query_port(ctx, port, &port_attr);

    pd = ibv_alloc_pd(ctx);
    ASSERT_NE(nullptr, pd);

    cq = ibv_create_cq(ctx, 8, NULL, NULL, 0);
    ASSERT_NE(nullptr, cq);
}

test_tl_mlx5::~test_tl_mlx5()
{
    if (cq) {
        ibv_destroy_cq(cq);
    }
    if (pd) {
        ibv_dealloc_pd(pd);
    }
    if (ctx) {
        ibv_close_device(ctx);
    }
    if (tl_mlx5_so_handle) {
        dlclose(tl_mlx5_so_handle);
    }
}

class test_tl_mlx5_rc_qp : public test_tl_mlx5 {
  public:
    ucc_tl_mlx5_qp_t qp;
    test_tl_mlx5_rc_qp()
    {
        qp.qp = NULL;
    }
    ~test_tl_mlx5_rc_qp()
    {
        if (qp.qp) {
            ibv_destroy_qp(qp.qp);
        }
    }
};

UCC_TEST_F(test_tl_mlx5_rc_qp, create)
{
    uint32_t     qpn;
    ucc_status_t status;

    status = create_rc_qp(ctx, pd, cq, port, 4, &qp, &qpn, &lib);
    EXPECT_EQ(UCC_OK, status);
}

UCC_TEST_F(test_tl_mlx5_rc_qp, connect)
{
    uint32_t     qpn;
    ucc_status_t status;

    status = create_rc_qp(ctx, pd, cq, port, 4, &qp, &qpn, &lib);
    EXPECT_EQ(UCC_OK, status);

    status = qp_connect(qp.qp, qpn, port_attr.lid, port, &lib);
    EXPECT_EQ(UCC_OK, status);
}

class test_tl_mlx5_dc : public test_tl_mlx5 {
  public:
    struct ibv_qp *   dct_qp;
    uint32_t          dct_qpn;
    struct ibv_srq *  srq;
    ucc_tl_mlx5_dci_t dci;
    struct ibv_ah *   ah;

    test_tl_mlx5_dc()
    {
        ah         = NULL;
        dct_qp     = NULL;
        dci.dci_qp = NULL;
        srq        = NULL;
    }
    virtual void SetUp()
    {
        struct ibv_srq_init_attr srq_attr;

        test_tl_mlx5::SetUp();
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

UCC_TEST_F(test_tl_mlx5_dc, init_dct)
{
    ucc_status_t status;

    status = init_dct(pd, ctx, cq, srq, port, &dct_qp, &dct_qpn, &lib);
    EXPECT_EQ(UCC_OK, status);
}

UCC_TEST_F(test_tl_mlx5_dc, init_dci)
{
    ucc_status_t status;

    status = init_dci(&dci, pd, ctx, cq, port, 4, &lib);
    EXPECT_EQ(UCC_OK, status);
}

UCC_TEST_F(test_tl_mlx5_dc, create_ah)
{
    ucc_status_t status;

    status = create_ah(pd, port_attr.lid, port, &ah, &lib);
    EXPECT_EQ(UCC_OK, status);
}
