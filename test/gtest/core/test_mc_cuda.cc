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
  protected:
    const int         TEST_ALLOC_SIZE = 1024;
    void *            test_ptr;
    ucc_memory_type_t test_mtype;
    virtual void SetUp() override
    {
        ucc::test::SetUp();
        ucc_constructor();
        ucc_mc_init();
        test_ptr   = NULL;
        test_mtype = UCC_MEMORY_TYPE_UNKNOWN;
    }
    virtual void TearDown() override
    {
        ucc_mc_finalize();
        ucc::test::TearDown();
    }
};

UCC_TEST_F(test_mc_cuda, mc_cuda_load)
{
    EXPECT_EQ(UCC_OK, ucc_mc_available(UCC_MEMORY_TYPE_CUDA));
}

UCC_TEST_F(test_mc_cuda, can_alloc_and_free_mem)
{
    EXPECT_EQ(UCC_OK,
              ucc_mc_alloc(&test_ptr, TEST_ALLOC_SIZE, UCC_MEMORY_TYPE_CUDA));
    EXPECT_EQ(cudaSuccess, cudaMemset(test_ptr, 0, TEST_ALLOC_SIZE));
    EXPECT_EQ(UCC_OK, ucc_mc_free(test_ptr, UCC_MEMORY_TYPE_CUDA));
}

UCC_TEST_F(test_mc_cuda, can_detect_host_mem)
{
    test_ptr = malloc(TEST_ALLOC_SIZE);
    if (test_ptr == NULL) {
        ADD_FAILURE() << "failed to allocate host memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_type(test_ptr, &test_mtype));
    EXPECT_EQ(UCC_MEMORY_TYPE_HOST, test_mtype);
    free(test_ptr);
}

UCC_TEST_F(test_mc_cuda, can_detect_cuda_mem)
{
    cudaError_t st;

    st = cudaMalloc(&test_ptr, TEST_ALLOC_SIZE);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate device memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_type(test_ptr, &test_mtype));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA, test_mtype);
    cudaFree(test_ptr);
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc_cuda, can_detect_managed_mem)
{
    cudaError_t st;

    st = cudaMallocManaged(&test_ptr, TEST_ALLOC_SIZE);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate managed memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_type(test_ptr, &test_mtype));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA_MANAGED, test_mtype);
    cudaFree(test_ptr);
}

UCC_TEST_F(test_mc_cuda, can_detect_host_alloc_mem)
{
    cudaError_t st;

    st = cudaHostAlloc(&test_ptr, TEST_ALLOC_SIZE, cudaHostAllocDefault);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate host mapped memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_type(test_ptr, &test_mtype));
    EXPECT_EQ(UCC_MEMORY_TYPE_HOST, test_mtype);
    cudaFreeHost(test_ptr);
}
