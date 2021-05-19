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
    const int               TEST_ALLOC_SIZE = 1024;
    ucc_mc_buffer_header_t *mc_header;
    void *                  test_ptr;
    ucc_memory_type_t       test_mtype;
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
              ucc_mc_alloc(&mc_header, TEST_ALLOC_SIZE, UCC_MEMORY_TYPE_CUDA));
    test_ptr = mc_header->addr;
    EXPECT_EQ(cudaSuccess, cudaMemset(test_ptr, 0, TEST_ALLOC_SIZE));
    EXPECT_EQ(UCC_OK, ucc_mc_free(mc_header, UCC_MEMORY_TYPE_CUDA));
}

// TODO: add UCC_TEST_F for multi threaded: spawn (multiple times - in a loop) pthreads and call ucc_mc_alloc/free.
// Make sure to allocate more than max amount of elems so it should test slow path as well

UCC_TEST_F(test_mc_cuda, can_detect_host_mem)
{
    ucc_mem_attr_t mem_attr;

    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    test_ptr = malloc(TEST_ALLOC_SIZE);
    if (test_ptr == NULL) {
        ADD_FAILURE() << "failed to allocate host memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_get_mem_attr(test_ptr, &mem_attr));
    EXPECT_EQ(UCC_MEMORY_TYPE_HOST, mem_attr.mem_type);
    free(test_ptr);
}

UCC_TEST_F(test_mc_cuda, can_detect_cuda_mem)
{
    ucc_mem_attr_t mem_attr;
    cudaError_t st;

    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    st = cudaMalloc(&test_ptr, TEST_ALLOC_SIZE);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate device memory";
    }
    EXPECT_EQ(UCC_OK, ucc_mc_get_mem_attr(test_ptr, &mem_attr));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA, mem_attr.mem_type);
    cudaFree(test_ptr);
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc_cuda, can_query_cuda_mem)
{
    ucc_mem_attr_t mem_attr;
    cudaError_t st;
    void *ptr;

    st = cudaMalloc(&test_ptr, TEST_ALLOC_SIZE);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate device memory";
    }

    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    EXPECT_EQ(UCC_OK, ucc_mc_get_mem_attr(test_ptr, &mem_attr));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA, mem_attr.mem_type);

    /* query base addr and length */
    mem_attr.field_mask   = UCC_MEM_ATTR_FIELD_MEM_TYPE |
                            UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                            UCC_MEM_ATTR_FIELD_ALLOC_LENGTH;
    mem_attr.alloc_length = 1;
    ptr = (char *)test_ptr + TEST_ALLOC_SIZE/2;
    EXPECT_EQ(UCC_OK, ucc_mc_get_mem_attr(ptr, &mem_attr));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA, mem_attr.mem_type);
    EXPECT_EQ(test_ptr, mem_attr.base_address);
    EXPECT_EQ(TEST_ALLOC_SIZE, mem_attr.alloc_length);

    cudaFree(test_ptr);
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc_cuda, can_detect_managed_mem)
{
    cudaError_t st;
    ucc_mem_attr_t mem_attr;

    st = cudaMallocManaged(&test_ptr, TEST_ALLOC_SIZE);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate managed memory";
    }
    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    EXPECT_EQ(UCC_OK, ucc_mc_get_mem_attr(test_ptr, &mem_attr));
    EXPECT_EQ(UCC_MEMORY_TYPE_CUDA_MANAGED, mem_attr.mem_type);
    cudaFree(test_ptr);
}

UCC_TEST_F(test_mc_cuda, can_detect_host_alloc_mem)
{
    cudaError_t st;
    ucc_mem_attr_t mem_attr;

    st = cudaHostAlloc(&test_ptr, TEST_ALLOC_SIZE, cudaHostAllocDefault);
    if (st != cudaSuccess) {
        ADD_FAILURE() << "failed to allocate host mapped memory";
    }
    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    EXPECT_EQ(UCC_OK, ucc_mc_get_mem_attr(test_ptr, &mem_attr));
    EXPECT_EQ(UCC_MEMORY_TYPE_HOST, mem_attr.mem_type);
    cudaFreeHost(test_ptr);
}
