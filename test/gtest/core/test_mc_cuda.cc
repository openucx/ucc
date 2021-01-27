/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include <common/test.h>
#include <cuda_runtime.h>

class test_mc_cuda : public ucc::test {
};

UCC_TEST_F(test_mc_cuda, mc_cuda_load)
{
    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());
    EXPECT_EQ(UCC_OK, ucc_mc_available(UCC_MEMORY_TYPE_CUDA));
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc_cuda, can_alloc_and_free_mem)
{
    void *ptr = NULL;
    size_t size = 4096;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());
    EXPECT_EQ(UCC_OK, ucc_mc_alloc(&ptr, size, UCC_MEMORY_TYPE_CUDA));
    EXPECT_EQ(cudaSuccess, cudaMemset(ptr, 0, size));
    EXPECT_EQ(UCC_OK, ucc_mc_free(ptr, UCC_MEMORY_TYPE_CUDA));
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc_cuda, can_detect_host_mem)
{
    void *host_ptr = NULL;
    size_t size = 4096;
    ucc_memory_type_t mtype;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());

    host_ptr = malloc(size);
    if (host_ptr == NULL) {
        ucc_mc_finalize();
        ADD_FAILURE() << "failed to allocate host memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_type(host_ptr, &mtype));
    EXPECT_EQ(UCC_MEMORY_TYPE_HOST, mtype);
    free(host_ptr);
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc_cuda, can_detect_cuda_mem)
{
    void *dev_ptr = NULL;
    size_t size = 4096;
    ucc_memory_type_t mtype;
    cudaError_t st;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());

    st = cudaMalloc(&dev_ptr, size);
    if (st != cudaSuccess) {
        ucc_mc_finalize();
        ADD_FAILURE() << "failed to allocate device memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_type(dev_ptr, &mtype));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA, mtype);
    cudaFree(dev_ptr);
    ucc_mc_finalize();
}
