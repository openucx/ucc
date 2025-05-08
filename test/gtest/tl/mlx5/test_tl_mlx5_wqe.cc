/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5_wqe.h"
#include "utils/arch/cpu.h"
#include <tuple>
#include <cmath>

// Rounds up a given integer to the closet power of two
static int roundUpToPowerOfTwo(int a)
{
    int b = 1;
    while (b < a) {
        b *= 2;
    }
    return b;
}

// Returns whether a matrix fits ConnectX-7 HW limitations
static bool doesMatrixFit(int nrows, int ncols, int elem_size)
{
    // Compute the matrix size as is done by ConnectX-7 HW
    int matrix_size =
        nrows * std::max(128, roundUpToPowerOfTwo(ncols) *
                                  roundUpToPowerOfTwo(std::max(elem_size, 8)));
    return matrix_size <= pow(2, 13) //= 8Kb
           && elem_size <= 128 && nrows <= 64 && ncols <= 64;
}

UCC_TEST_P(test_tl_mlx5_transpose, transposeWqe)
{
    int            nrows           = std::get<0>(GetParam());
    int            ncols           = std::get<1>(GetParam());
    int            elem_size       = sizeof(DT) * std::get<2>(GetParam());
    int            completions_num = 0;
    DT             src[nrows][ncols * elem_size], dst[ncols][nrows * elem_size];
    struct ibv_wc  wcs[1];
    struct ibv_mr *src_mr, *dst_mr;
    int            i, j, k;
    struct ibv_device_attr device_attr;

    GTEST_ASSERT_EQ(ibv_query_device(ctx, &device_attr), 0);
    // Check for Mellanox/NVIDIA vendor ID (0x02c9) and CX7 (MT4129) vendor_part_id
    if (device_attr.vendor_id != 0x02c9 || device_attr.vendor_part_id != 4129) {
        GTEST_SKIP() << "The test needs CX7 but got vendor_id="
                     << device_attr.vendor_id
                     << ", vendor_part_id=" << device_attr.vendor_part_id;
    }

    // Skips if do not match HW limitations
    if (!doesMatrixFit(nrows, ncols, elem_size)) {
        GTEST_SKIP();
    }

    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncols; j++) {
            for (k = 0; k < elem_size; k++) {
                src[i][j * elem_size + k] =
                    (i * nrows * elem_size + j * elem_size + k) % 256;
                dst[j][i * elem_size + k] = 0;
            }
        }
    }

    src_mr =
        ibv_reg_mr(pd, src, nrows * ncols * elem_size, IBV_ACCESS_LOCAL_WRITE);
    GTEST_ASSERT_NE(nullptr, src_mr);
    dst_mr =
        ibv_reg_mr(pd, dst, nrows * ncols * elem_size, IBV_ACCESS_LOCAL_WRITE);
    GTEST_ASSERT_NE(nullptr, dst_mr);

    ibv_wr_start(qp.qp_ex);
    post_transpose(qp.qp, src_mr->lkey, dst_mr->rkey, (uintptr_t)src,
                   (uintptr_t)dst, elem_size, ncols, nrows, IBV_SEND_SIGNALED);
    GTEST_ASSERT_EQ(ibv_wr_complete(qp.qp_ex), 0);

    while (!completions_num) {
        completions_num = ibv_poll_cq(cq, 1, wcs);
    }
    GTEST_ASSERT_EQ(completions_num, 1);
    GTEST_ASSERT_EQ(wcs[0].status, IBV_WC_SUCCESS);

    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ncols; j++) {
            for (k = 0; k < elem_size; k++) {
                GTEST_ASSERT_EQ(src[i][j * elem_size + k],
                                dst[j][i * elem_size + k]);
            }
        }
    }

    GTEST_ASSERT_EQ(ibv_dereg_mr(src_mr), UCC_OK);
    GTEST_ASSERT_EQ(ibv_dereg_mr(dst_mr), UCC_OK);
}

INSTANTIATE_TEST_SUITE_P(, test_tl_mlx5_transpose,
                         ::testing::Combine(::testing::Values(1, 7, 32, 64),
                                            ::testing::Values(1, 5, 32, 64),
                                            ::testing::Values(1, 3, 8, 128)));

UCC_TEST_P(test_tl_mlx5_rdma_write, RdmaWriteWqe)
{
    struct ibv_sge     sg;
    struct ibv_send_wr wr;

    bufsize = GetParam();
    buffers_init();
    CHECK_TEST_STATUS();

    memset(&sg, 0, sizeof(sg));
    sg.addr   = (uintptr_t)src;
    sg.length = bufsize;
    sg.lkey   = src_mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id               = 0;
    wr.sg_list             = &sg;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE;
    wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    wr.next                = NULL;
    wr.wr.rdma.remote_addr = (uintptr_t)dst;
    wr.wr.rdma.rkey        = dst_mr->rkey;

    // This work request is posted with wr_id = 0
    GTEST_ASSERT_EQ(ibv_post_send(qp.qp, &wr, NULL), 0);
    wait_for_completion();
    CHECK_TEST_STATUS();

    validate_buffers();
}

UCC_TEST_P(test_tl_mlx5_rdma_write, CustomRdmaWriteWqe)
{
    bufsize = GetParam();
    buffers_init();
    CHECK_TEST_STATUS();

    ibv_wr_start(qp.qp_ex);
    post_rdma_write(qp.qp, qpn, nullptr, (uintptr_t)src, bufsize, src_mr->lkey,
                    (uintptr_t)dst, dst_mr->rkey,
                    IBV_SEND_SIGNALED | IBV_SEND_FENCE, 0);
    GTEST_ASSERT_EQ(ibv_wr_complete(qp.qp_ex), 0);
    wait_for_completion();
    CHECK_TEST_STATUS();

    validate_buffers();
}

INSTANTIATE_TEST_SUITE_P(, test_tl_mlx5_rdma_write,
                         ::testing::Values(1, 31, 128, 1024));

UCC_TEST_P(test_tl_mlx5_dm, MemcpyToDeviceMemory)
{
    bufsize = GetParam();
    buffers_init();
    CHECK_TEST_STATUS();
    if (!dm_ptr) {
        return;
    }

    if (bufsize % 4 != 0) {
        GTEST_SKIP() << "for memcpy involving device memory, buffer size "
                     << "must be a multiple of 4";
    }

    GTEST_ASSERT_EQ(ibv_memcpy_to_dm(dm_ptr, 0, (void *)src, bufsize), 0);
    GTEST_ASSERT_EQ(ibv_memcpy_from_dm((void *)dst, dm_ptr, 0, bufsize), 0);

    validate_buffers();
}

UCC_TEST_P(test_tl_mlx5_dm, RdmaToDeviceMemory)
{
    struct ibv_sge     sg;
    struct ibv_send_wr wr;

    bufsize = GetParam();
    buffers_init();
    CHECK_TEST_STATUS();
    if (!dm_ptr) {
        return;
    }

    // RDMA write from host source to device memory
    memset(&sg, 0, sizeof(sg));
    sg.addr   = (uintptr_t)src;
    sg.length = bufsize;
    sg.lkey   = src_mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id               = 0;
    wr.sg_list             = &sg;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE;
    wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    wr.next                = NULL;
    wr.wr.rdma.remote_addr = (uintptr_t)0;
    wr.wr.rdma.rkey        = dm_mr->rkey;

    GTEST_ASSERT_EQ(ibv_post_send(qp.qp, &wr, NULL), 0);
    wait_for_completion();
    CHECK_TEST_STATUS();

    // RDMA write from device memory to host destination
    memset(&sg, 0, sizeof(sg));
    sg.addr   = (uintptr_t)0;
    sg.length = bufsize;
    sg.lkey   = dm_mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id               = 0;
    wr.sg_list             = &sg;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE;
    wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    wr.next                = NULL;
    wr.wr.rdma.remote_addr = (uintptr_t)dst;
    wr.wr.rdma.rkey        = dst_mr->rkey;

    GTEST_ASSERT_EQ(ibv_post_send(qp.qp, &wr, NULL), 0);
    wait_for_completion();
    CHECK_TEST_STATUS();

    validate_buffers();
}

UCC_TEST_P(test_tl_mlx5_dm, CustomRdmaToDeviceMemory)
{
    bufsize = GetParam();
    buffers_init();
    CHECK_TEST_STATUS();
    if (!dm_ptr) {
        return;
    }

    // RDMA write from host source to device memory
    ibv_wr_start(qp.qp_ex);
    post_rdma_write(qp.qp, qpn, nullptr, (uintptr_t)src, bufsize, src_mr->lkey,
                    (uintptr_t)0, dm_mr->rkey,
                    IBV_SEND_SIGNALED | IBV_SEND_FENCE, 0);
    GTEST_ASSERT_EQ(ibv_wr_complete(qp.qp_ex), 0);
    wait_for_completion();
    CHECK_TEST_STATUS();

    // RDMA write from device memory to host destination
    ibv_wr_start(qp.qp_ex);
    post_rdma_write(qp.qp, qpn, nullptr, (uintptr_t)0, bufsize, dm_mr->lkey,
                    (uintptr_t)dst, dst_mr->rkey,
                    IBV_SEND_SIGNALED | IBV_SEND_FENCE, 0);
    GTEST_ASSERT_EQ(ibv_wr_complete(qp.qp_ex), 0);
    wait_for_completion();
    CHECK_TEST_STATUS();

    validate_buffers();
}

INSTANTIATE_TEST_SUITE_P(, test_tl_mlx5_dm,
                         ::testing::Values(1, 12, 31, 32, 8192, 8193, 32768,
                                           65536));

UCC_TEST_P(test_tl_mlx5_wait_on_data, waitOnDataWqe)
{
    uint64_t           wait_on_value   = std::get<0>(GetParam());
    uint64_t           init_ctrl_value = std::get<1>(GetParam());
    uint64_t           buffer[3];
    volatile uint64_t *ctrl, *src, *dst;
    int                completions_num;
    struct ibv_wc      wcs[1];
    struct ibv_mr *    buffer_mr;
    struct ibv_sge     sg;
    struct ibv_send_wr wr;

    memset(buffer, 0, 3 * sizeof(uint64_t));
    buffer_mr = ibv_reg_mr(pd, buffer, 3 * sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    GTEST_ASSERT_NE(nullptr, buffer_mr);
    ctrl = &buffer[0];
    src  = &buffer[1];
    dst  = &buffer[2];

    *ctrl = init_ctrl_value;

    memset(&sg, 0, sizeof(sg));
    sg.addr   = (uintptr_t)src;
    sg.length = sizeof(uint64_t);
    sg.lkey   = buffer_mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id               = 0;
    wr.sg_list             = &sg;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE;
    wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_FENCE;
    wr.next                = NULL;
    wr.wr.rdma.remote_addr = (uintptr_t)dst;
    wr.wr.rdma.rkey        = buffer_mr->rkey;

    // This work request is posted with wr_id = 1
    GTEST_ASSERT_EQ(post_wait_on_data(qp.qp, wait_on_value, buffer_mr->lkey,
                                      (uintptr_t)ctrl, nullptr),
                    UCC_OK);
    // This work request is posted with wr_id = 0
    GTEST_ASSERT_EQ(ibv_post_send(qp.qp, &wr, NULL), 0);

    sleep(1);

    *src = 0xdeadbeef;
    //memory barrier
    ucc_memory_cpu_fence();
    *ctrl = wait_on_value;

    while (1) {
        completions_num = ibv_poll_cq(cq, 1, wcs);
        if (completions_num != 0) {
            GTEST_ASSERT_EQ(completions_num, 1);
            GTEST_ASSERT_EQ(wcs[0].status, IBV_WC_SUCCESS);
            if (wcs[0].wr_id == 0) {
                break;
            }
        }
    }

    //validation
    GTEST_ASSERT_EQ(*dst, *src);

    GTEST_ASSERT_EQ(ibv_dereg_mr(buffer_mr), UCC_OK);
}

INSTANTIATE_TEST_SUITE_P(
    , test_tl_mlx5_wait_on_data,
    ::testing::Combine(::testing::Values(1, 1024, 1025, 0xF0F30F00, 0xFFFFFFFF),
                       ::testing::Values(0, 0xF0F30F01)));

UCC_TEST_P(test_tl_mlx5_umr_wqe, umrWqe)
{
    uint16_t nbr_srcs              = std::get<0>(GetParam());
    uint32_t bytes_count           = std::get<1>(GetParam());
    uint32_t repeat_count          = std::get<2>(GetParam());
    uint32_t bytes_skip            = std::get<3>(GetParam());
    int      src_size              = (bytes_count + bytes_skip) * repeat_count;
    int      dst_size              = bytes_count * nbr_srcs * repeat_count;
    int      send_mem_access_flags = 0;
    int      recv_mem_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    DT                           src[nbr_srcs][src_size], dst[dst_size];
    struct ibv_mr *              src_mr[nbr_srcs], *dst_mr, *umr_entries_mr;
    struct mlx5dv_mkey *         umr_mkey;
    struct mlx5dv_mkey_init_attr umr_mkey_init_attr;
    void *                       umr_entries_buf;
    size_t                       umr_buf_size;
    struct mlx5dv_mr_interleaved mkey_entries[nbr_srcs];
    struct ibv_wc                wcs[1];
    int i, j, completions_num, src_index, repetition_count, offset;

    // Setup src and dst buffers
    for (i = 0; i < nbr_srcs; i++) {
        for (j = 0; j < src_size; j++) {
            src[i][j] = (i * (src_size + 7) + j) % 255;
        }
        src_mr[i] = ibv_reg_mr(pd, src[i], src_size, send_mem_access_flags);
        GTEST_ASSERT_NE(nullptr, src_mr[i]);
    }

    memset(dst, 0, dst_size);
    dst_mr = ibv_reg_mr(pd, dst, dst_size, recv_mem_access_flags);
    GTEST_ASSERT_NE(nullptr, dst_mr);

    // UMR
    umr_buf_size = ucc_align_up(
        sizeof(struct mlx5_wqe_umr_repeat_ent_seg) * (nbr_srcs + 1), 64);
    GTEST_ASSERT_EQ(ucc_posix_memalign(&umr_entries_buf, 2048, umr_buf_size),
                    0);

    umr_entries_mr =
        ibv_reg_mr(pd, umr_entries_buf, umr_buf_size, send_mem_access_flags);
    GTEST_ASSERT_NE(nullptr, umr_entries_mr);

    memset(&umr_mkey_init_attr, 0, sizeof(umr_mkey_init_attr));
    umr_mkey_init_attr.pd           = pd;
    umr_mkey_init_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    umr_mkey_init_attr.max_entries  = nbr_srcs + 1; //+1 for the "repeat block"

    umr_mkey = mlx5dv_create_mkey(&umr_mkey_init_attr);
    GTEST_ASSERT_NE(nullptr, umr_mkey);
    GTEST_ASSERT_GE(umr_mkey_init_attr.max_entries, nbr_srcs + 1);

    for (i = 0; i < nbr_srcs; i++) {
        mkey_entries[i].addr        = (uintptr_t)src[i];
        mkey_entries[i].bytes_count = bytes_count;
        mkey_entries[i].bytes_skip  = bytes_skip;
        mkey_entries[i].lkey        = src_mr[i]->lkey;
    }

    post_umr(umr_qp.qp, umr_mkey, send_mem_access_flags, repeat_count, nbr_srcs,
             mkey_entries, (uint32_t)umr_entries_mr->lkey, umr_entries_buf);

    completions_num = 0;
    while (!completions_num) {
        completions_num = ibv_poll_cq(cq, 1, wcs);
    }
    GTEST_ASSERT_EQ(completions_num, 1);
    GTEST_ASSERT_EQ(wcs[0].status, IBV_WC_SUCCESS);
    GTEST_ASSERT_EQ(wcs[0].wr_id, 0);

    // RDMA Write
    ibv_wr_start(qp.qp_ex);
    post_rdma_write(qp.qp, qpn, nullptr, (uintptr_t)0, dst_size, umr_mkey->lkey,
                    (uintptr_t)dst, dst_mr->rkey,
                    IBV_SEND_SIGNALED | IBV_SEND_FENCE, 0);
    GTEST_ASSERT_EQ(ibv_wr_complete(qp.qp_ex), 0);

    completions_num = 0;
    while (!completions_num) {
        completions_num = ibv_poll_cq(cq, 1, wcs);
    }
    GTEST_ASSERT_EQ(completions_num, 1);
    GTEST_ASSERT_EQ(wcs[0].status, IBV_WC_SUCCESS);
    GTEST_ASSERT_EQ(wcs[0].wr_id, 0);

    // Verification
    for (i = 0; i < dst_size; i++) {
        src_index        = (i / bytes_count) % nbr_srcs;
        repetition_count = (i / bytes_count) / nbr_srcs;
        offset =
            repetition_count * (bytes_count + bytes_skip) + (i % bytes_count);
        GTEST_ASSERT_EQ(dst[i], src[src_index][offset]);
    }

    // Tear down
    GTEST_ASSERT_EQ(0, mlx5dv_destroy_mkey(umr_mkey));
    GTEST_ASSERT_EQ(ibv_dereg_mr(umr_entries_mr), UCC_OK);
    ucc_free(umr_entries_buf);
    for (i = 0; i < nbr_srcs; i++) {
        GTEST_ASSERT_EQ(ibv_dereg_mr(src_mr[i]), UCC_OK);
    }
    GTEST_ASSERT_EQ(ibv_dereg_mr(dst_mr), UCC_OK);
}

INSTANTIATE_TEST_SUITE_P(, test_tl_mlx5_umr_wqe,
                         ::testing::Combine(::testing::Values(1, 129, 1024),
                                            ::testing::Values(5, 64),
                                            ::testing::Values(1, 3, 16),
                                            ::testing::Values(0, 7)));

UCC_TEST_P(test_tl_mlx5_dm_alloc_reg, DeviceMemoryAllocation)
{
    size_t         buf_size = std::get<0>(GetParam());
    size_t         buf_num  = std::get<1>(GetParam());
    struct ibv_dm *ptr      = nullptr;
    struct ibv_mr *mr       = nullptr;
    ucc_status_t   status;

    status = dm_alloc_reg(ctx, pd, 0, buf_size, &buf_num, &ptr, &mr, &lib);
    if (status == UCC_ERR_NO_MEMORY || status == UCC_ERR_NO_RESOURCE) {
        GTEST_SKIP() << "cannot allocate " << buf_num << " chunk(s) of size "
                     << buf_size << " in device memory";
    }
    GTEST_ASSERT_EQ(status, UCC_OK);

    ibv_dereg_mr(mr);
    ibv_free_dm(ptr);
}

INSTANTIATE_TEST_SUITE_P(
    , test_tl_mlx5_dm_alloc_reg,
    ::testing::Combine(::testing::Values(1, 2, 1024, 8191, 8192, 8193, 32768,
                                         65536, 262144),
                       ::testing::Values(UCC_ULUNITS_AUTO, 1, 3, 8)));
