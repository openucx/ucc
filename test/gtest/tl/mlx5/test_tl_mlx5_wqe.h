/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5.h"
#include "test_tl_mlx5_qps.h"
#include "components/tl/mlx5/tl_mlx5_wqe.h"

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

class test_tl_mlx5_rdma_write
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<RdmaWriteParams> {
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
