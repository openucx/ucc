/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5.h"
#include "test_tl_mlx5_qps.h"
#include "components/tl/mlx5/tl_mlx5_wqe.h"

#define DT uint8_t

typedef ucc_status_t (*ucc_tl_mlx5_post_rdma_fn_t)(
    struct ibv_qp *qp, uint32_t qpn, struct ibv_ah *ah, uintptr_t src_mkey_addr,
    size_t len, uint32_t src_mr_lkey, uintptr_t dst_addr, uint32_t dst_mr_key,
    int send_flags, uint64_t wr_id);
typedef ucc_status_t (*ucc_tl_mlx5_post_transpose_fn_t)(
    struct ibv_qp *qp, uint32_t src_mr_lkey, uint32_t dst_mr_key,
    uintptr_t src_mkey_addr, uintptr_t dst_addr, uint32_t element_size,
    uint16_t ncols, uint16_t nrows, int send_flags);

typedef ucc_status_t (*ucc_tl_mlx5_post_wait_on_data_fn_t)(struct ibv_qp *qp,
                                                           uint64_t       value,
                                                           uint32_t       lkey,
                                                           uintptr_t      addr,
                                                           void *task_ptr);

typedef ucc_status_t (*ucc_tl_mlx5_post_umr_fn_t)(
    struct ibv_qp *qp, struct mlx5dv_mkey *dv_mkey, uint32_t access_flags,
    uint32_t repeat_count, uint16_t num_entries,
    struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey, void *ptr_address);

//    (msgsize)
using RdmaWriteParams = int;
//    (nrows, ncols, element_size)
using TransposeParams = std::tuple<int, int, int>;
//    (nbr_srcs, bytes_count, repeat_count, bytes_skip)
using UmrParams = std::tuple<int, int, int, int>;

class test_tl_mlx5_wqe : public test_tl_mlx5_rc_qp {
  public:
    ucc_tl_mlx5_post_rdma_fn_t         post_rdma_write;
    ucc_tl_mlx5_post_transpose_fn_t    post_transpose;
    ucc_tl_mlx5_post_wait_on_data_fn_t post_wait_on_data;
    ucc_tl_mlx5_post_umr_fn_t          post_umr;

    void SetUp()
    {
        test_tl_mlx5_rc_qp::SetUp();

        post_rdma_write = (ucc_tl_mlx5_post_rdma_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_post_rdma");
        ASSERT_EQ(nullptr, dlerror());

        post_transpose = (ucc_tl_mlx5_post_transpose_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_post_transpose");
        ASSERT_EQ(nullptr, dlerror());

        post_wait_on_data = (ucc_tl_mlx5_post_wait_on_data_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_post_wait_on_data");
        ASSERT_EQ(nullptr, dlerror());

        post_umr = (ucc_tl_mlx5_post_umr_fn_t)dlsym(tl_mlx5_so_handle,
                                                    "ucc_tl_mlx5_post_umr");
        ASSERT_EQ(nullptr, dlerror());

        create_qp();
        connect_qp_loopback();
        create_umr_qp();
    }
};

class test_tl_mlx5_transpose
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<TransposeParams> {
};

class test_tl_mlx5_wait_on_data : public test_tl_mlx5_wqe {
};

class test_tl_mlx5_umr_wqe : public test_tl_mlx5_wqe,
                             public ::testing::WithParamInterface<UmrParams> {
};

class test_tl_mlx5_rdma_write
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<RdmaWriteParams> {
public:
    int bufsize;
    DT *src, *dst;
    struct ibv_mr *src_mr, *dst_mr;

    void buffers_init()
    {
        src = (DT*) malloc(bufsize);
        GTEST_ASSERT_NE(src, nullptr);
        dst = (DT*) malloc(bufsize);
        GTEST_ASSERT_NE(dst, nullptr);

        for (int i = 0; i < bufsize; i++) {
            src[i] = i % 256;
            dst[i] = 0;
        }

        src_mr = ibv_reg_mr(pd, src, bufsize,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        GTEST_ASSERT_NE(nullptr, src_mr);
        dst_mr = ibv_reg_mr(pd, dst, bufsize,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        GTEST_ASSERT_NE(nullptr, dst_mr);

    }

    void wait_for_completion()
    {
        int completions_num = 0;
        struct ibv_wc  wcs[1];

        while (!completions_num) {
            completions_num = ibv_poll_cq(cq, 1, wcs);
        }

        GTEST_ASSERT_EQ(completions_num, 1);
        GTEST_ASSERT_EQ(wcs[0].status, IBV_WC_SUCCESS);
    }

    void validate_buffers()
    {
        for (int i = 0; i < bufsize; i++) {
            GTEST_ASSERT_EQ(src[i], dst[i]);
        }
    }

    void TearDown()
    {
        GTEST_ASSERT_EQ(ibv_dereg_mr(src_mr), UCC_OK);
        GTEST_ASSERT_EQ(ibv_dereg_mr(dst_mr), UCC_OK);
        free(src);
        free(dst);
    }
};

