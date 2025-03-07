/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5_qps.h"

UCC_TEST_F(test_tl_mlx5_rc_qp, create)
{
    create_qp();
}

UCC_TEST_F(test_tl_mlx5_rc_qp, create_umr)
{
    create_umr_qp();
}

UCC_TEST_F(test_tl_mlx5_rc_qp, connect_loopback)
{
    create_qp();
    CHECK_TEST_STATUS();
    connect_qp_loopback();
}

UCC_TEST_F(test_tl_mlx5_dc, init_dct)
{
    ucc_status_t status;

    status =
        init_dct(pd, ctx, cq, srq, port, &dct_qp, &dct_qpn, &qp_conf, &lib);
    GTEST_ASSERT_EQ(UCC_OK, status);
}

UCC_TEST_F(test_tl_mlx5_dc, init_dci)
{
    ucc_status_t status;

    status = init_dci(&dci, pd, ctx, cq, port, 4, &qp_conf, &lib);
    GTEST_ASSERT_EQ(UCC_OK, status);
}

UCC_TEST_F(test_tl_mlx5_dc, create_ah)
{
    ucc_status_t status;

    status = create_ah(pd, port_attr.lid, port, &ah, &lib);
    GTEST_ASSERT_EQ(UCC_OK, status);
}
