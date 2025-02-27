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

typedef ucc_status_t (*ucc_tl_mlx5_dm_alloc_reg_fn_t)(
    struct ibv_context *ib_ctx, struct ibv_pd *pd, int dm_host, size_t buf_size,
    size_t *buf_num_p, struct ibv_dm **ptr, struct ibv_mr **mr,
    ucc_base_lib_t *lib);

//    (msgsize)
using RdmaWriteParams = int;
//    (buf_size)
using DmParams = int;
//    (nrows, ncols, element_size)
using TransposeParams = std::tuple<int, int, int>;
//    (nbr_srcs, bytes_count, repeat_count, bytes_skip)
using UmrParams = std::tuple<int, int, int, int>;
//    (buffer_size, buffer_nums)
using AllocDmParams = std::tuple<int, int>;
//    (wait_on_value, init_ctrl_value)
using WaitOnDataParams = std::tuple<uint64_t, uint64_t>;

class test_tl_mlx5_wqe : public test_tl_mlx5_rc_qp {
  public:
    ucc_tl_mlx5_post_rdma_fn_t         post_rdma_write;
    ucc_tl_mlx5_post_transpose_fn_t    post_transpose;
    ucc_tl_mlx5_post_wait_on_data_fn_t post_wait_on_data;
    ucc_tl_mlx5_post_umr_fn_t          post_umr;

    void SetUp()
    {
        test_tl_mlx5_rc_qp::SetUp();
        CHECK_TEST_STATUS();

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
        CHECK_TEST_STATUS();
        connect_qp_loopback();
        CHECK_TEST_STATUS();
        create_umr_qp();
    }
};

class test_tl_mlx5_transpose
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<TransposeParams> {
};

class test_tl_mlx5_wait_on_data
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<WaitOnDataParams> {
};

class test_tl_mlx5_umr_wqe : public test_tl_mlx5_wqe,
                             public ::testing::WithParamInterface<UmrParams> {
};

class test_tl_mlx5_rdma_write
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<RdmaWriteParams> {
  public:
    DT            *src    = nullptr;
    DT            *dst    = nullptr;
    struct ibv_mr *src_mr = nullptr;
    struct ibv_mr *dst_mr = nullptr;
    int            bufsize;

    void buffers_init()
    {
        src = (DT *)malloc(bufsize);
        GTEST_ASSERT_NE(src, nullptr);
        dst = (DT *)malloc(bufsize);
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
        int           completions_num = 0;
        struct ibv_wc wcs[1];

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
        if (src_mr != nullptr) {
            GTEST_ASSERT_EQ(ibv_dereg_mr(src_mr), UCC_OK);
        }
        if (dst_mr != nullptr) {
            GTEST_ASSERT_EQ(ibv_dereg_mr(dst_mr), UCC_OK);
        }
        if (src != nullptr) {
            free(src);
        }
        if (dst != nullptr) {
            free(dst);
        }
    }
};

class test_tl_mlx5_dm : public test_tl_mlx5_rdma_write {
  public:
    struct ibv_dm *dm_ptr = nullptr;
    struct ibv_mr *dm_mr  = nullptr;
    struct ibv_alloc_dm_attr dm_attr;

    void buffers_init()
    {
        test_tl_mlx5_rdma_write::buffers_init();
        CHECK_TEST_STATUS();

        struct ibv_device_attr_ex attr;
        memset(&attr, 0, sizeof(attr));
        GTEST_ASSERT_EQ(ibv_query_device_ex(ctx, NULL, &attr), 0);
        if (attr.max_dm_size < bufsize) {
            if (!attr.max_dm_size) {
                GTEST_SKIP() << "device doesn't support dm allocation";
            } else {
                GTEST_SKIP() << "the requested buffer size (=" << bufsize
                             << ") for device memory should be less than "
                             << attr.max_dm_size;
            }
        }

        memset(&dm_attr, 0, sizeof(dm_attr));
        dm_attr.length = bufsize;
        dm_ptr         = ibv_alloc_dm(ctx, &dm_attr);
        if (!dm_ptr) {
            GTEST_SKIP() << "device cannot allocate a buffer of size "
                         << bufsize;
        }

        dm_mr = ibv_reg_dm_mr(pd, dm_ptr, 0, dm_attr.length,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  IBV_ACCESS_ZERO_BASED);
        GTEST_ASSERT_NE(dm_mr, nullptr);
    }

    void TearDown()
    {
        if (dm_mr) {
            ibv_dereg_mr(dm_mr);
        }
        if (dm_ptr) {
            ibv_free_dm(dm_ptr);
        }
        test_tl_mlx5_rdma_write::TearDown();
    }
};

class test_tl_mlx5_dm_alloc_reg
    : public test_tl_mlx5_wqe,
      public ::testing::WithParamInterface<AllocDmParams> {
  public:
    ucc_tl_mlx5_dm_alloc_reg_fn_t dm_alloc_reg;
    void SetUp()
    {
        test_tl_mlx5_wqe::SetUp();
        CHECK_TEST_STATUS();

        dm_alloc_reg = (ucc_tl_mlx5_dm_alloc_reg_fn_t)dlsym(
            tl_mlx5_so_handle, "ucc_tl_mlx5_dm_alloc_reg");
        ASSERT_EQ(nullptr, dlerror());
    }
};
