/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include <components/ec/ucc_ec.h>
#include <pthread.h>
}
#include <common/test.h>
#include <common/test_ucc.h>
#include <cuda_runtime.h>

class test_ec_cuda : public ucc::test {
    void TestECCudaSetUp(ucc_ec_params_t ec_params)
    {
        ucc::test::SetUp();
        ucc_constructor();
        ucc_ec_init(&ec_params);
    }

    virtual void SetUp() override
    {
        ucc_ec_params_t ec_params = {
            .thread_mode = UCC_THREAD_SINGLE,
        };

        TestECCudaSetUp(ec_params);
        if (UCC_OK != ucc_ec_available(UCC_EE_CUDA_STREAM)) {
            GTEST_SKIP();
        }
    }

    virtual void TearDown() override
    {
        ucc_ec_finalize();
        ucc::test::TearDown();
    }

public:
    ucc_status_t get_cuda_executor(ucc_ee_executor_t **executor)
    {
        ucc_ee_executor_params_t eparams;

        eparams.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
        eparams.ee_type = UCC_EE_CUDA_STREAM;

        return ucc_ee_executor_init(&eparams, executor);
    }

    ucc_status_t put_cuda_executor(ucc_ee_executor_t *executor)
    {
        return ucc_ee_executor_finalize(executor);
    }

};

UCC_TEST_F(test_ec_cuda, ec_cuda_load)
{
    ASSERT_EQ(UCC_OK, ucc_ec_available(UCC_EE_CUDA_STREAM));
}

UCC_TEST_F(test_ec_cuda, ec_cuda_executor_init_finalize)
{
    ucc_ee_executor_t *executor;

    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}

UCC_TEST_F(test_ec_cuda, ec_cuda_executor_interruptible_start)
{
    ucc_ee_executor_t *executor;
    ucc_status_t status;

    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));

    status = ucc_ee_executor_start(executor, nullptr);
    EXPECT_EQ(UCC_OK, status);
    if (status == UCC_OK) {
        EXPECT_EQ(UCC_OK, ucc_ee_executor_stop(executor));
    }
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}


UCC_TEST_F(test_ec_cuda, ec_cuda_executor_interruptible_copy)
{
    ucc_ee_executor_t *executor;
    ucc_status_t status;
    cudaError_t cuda_st;
    int *src_host, *dst_host, *src, *dst;
    const size_t size = 4850;
    ucc_ee_executor_task_t *task;

    ASSERT_EQ(UCC_OK, get_cuda_executor(&executor));

    status = ucc_ee_executor_start(executor, nullptr);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    src_host = (int*)malloc(size * sizeof(int));
    EXPECT_NE(nullptr, src_host);
    if (!src_host) {
        goto exit;
    }

    dst_host = (int*)malloc(size * sizeof(int));
    EXPECT_NE(nullptr, dst_host);
    if (!dst_host) {
        goto exit;
    }

    cuda_st = cudaMalloc(&src, size * sizeof(int));
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    cuda_st = cudaMalloc(&dst, size * sizeof(int));
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    for (int i = 0; i < size; i++) {
        src_host[i] = i;
    }

    cuda_st = cudaMemcpy(src, src_host, size * sizeof(int), cudaMemcpyDefault);
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    ucc_ee_executor_task_args_t args;
    args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    args.copy.src  = src;
    args.copy.dst  = dst;
    args.copy.len  = size * sizeof(int);

    status = ucc_ee_executor_task_post(executor, &args, &task);
    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    do {
        status = ucc_ee_executor_task_test(task);
    } while (status > 0);

    EXPECT_EQ(UCC_OK, status);
    if (status != UCC_OK) {
        goto exit;
    }

    cuda_st = cudaMemcpy(dst_host, dst, size * sizeof(int), cudaMemcpyDefault);
    EXPECT_EQ(cudaSuccess, cuda_st);
    if (cuda_st != cudaSuccess) {
        goto exit;
    }

    for (int i = 0; i < size; i++) {
        if (dst_host[i] != src_host[i]) {
            EXPECT_EQ(dst_host[i], src_host[i]);
            goto exit;
        }
    }

exit:
    if (task) {
        EXPECT_EQ(UCC_OK, ucc_ee_executor_task_finalize(task));
    }
    if (src_host) {
        free(src_host);
    }

    if (dst_host) {
        free(dst_host);
    }

    if (src) {
        cudaFree(src);
    }

    if (dst) {
        cudaFree(dst);
    }
    EXPECT_EQ(UCC_OK, ucc_ee_executor_stop(executor));
    ASSERT_EQ(UCC_OK, put_cuda_executor(executor));
}
