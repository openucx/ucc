/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "test_context.h"
#include "../common/test_ucc.h"
#include <vector>
#include <algorithm>
#include <random>

test_context::test_context()
{
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));
}

test_context::~test_context()
{
    ucc_context_config_release(ctx_config);
}

UCC_TEST_F(test_context, create_destroy)
{
    ucc_context_params_t ctx_params;
    ucc_context_h        ctx_h;
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE;
    ctx_params.type = UCC_CONTEXT_EXCLUSIVE;
    EXPECT_EQ(UCC_OK, ucc_context_create(lib_h, &ctx_params, ctx_config, &ctx_h));
    EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
}

UCC_TEST_F(test_context, init_multiple)
{
    const int                  n_ctxs = 8;
    ucc_context_params_t       ctx_params;
    ucc_context_h              ctx_h;
    std::vector<ucc_context_h> ctxs;
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE;
    ctx_params.type = UCC_CONTEXT_EXCLUSIVE;
    for (int i = 0; i < n_ctxs; i++) {
        EXPECT_EQ(UCC_OK, ucc_context_create(lib_h, &ctx_params, ctx_config, &ctx_h));
        ctxs.push_back(ctx_h);
    }

    std::shuffle(ctxs.begin(), ctxs.end(), std::default_random_engine());
    for (auto ctx_h : ctxs) {
        EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
    }
}

test_context_get_attr::test_context_get_attr()
{
    ucc_context_params_t ctx_params;
    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE;
    ctx_params.type = UCC_CONTEXT_EXCLUSIVE;
    EXPECT_EQ(UCC_OK,
              ucc_context_create(lib_h, &ctx_params, ctx_config, &ctx_h));
}

test_context_get_attr::~test_context_get_attr()
{
    EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
}

UCC_TEST_F(test_context_get_attr, addr_len)
{
    ucc_context_attr_t attr;
    attr.mask = UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN;
    EXPECT_EQ(UCC_OK, ucc_context_get_attr(ctx_h, &attr));
}

UCC_TEST_F(test_context_get_attr, addr)
{
    ucc_context_attr_t attr;
    attr.mask = UCC_CONTEXT_ATTR_FIELD_CTX_ADDR;
    EXPECT_EQ(UCC_OK, ucc_context_get_attr(ctx_h, &attr));
    EXPECT_EQ(true, ((attr.ctx_addr_len == 0) || (NULL != attr.ctx_addr)));
}

UCC_TEST_F(test_context_get_attr, work_buffer_size)
{
    ucc_context_attr_t attr;
    attr.mask = UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE;
    EXPECT_EQ(UCC_OK, ucc_context_get_attr(ctx_h, &attr));
    EXPECT_EQ(5, attr.global_work_buffer_size);
}

UCC_TEST_F(test_context, global)
{
    /* Create and cleanup several Jobs (ucc contextss) with OOB */
    UccJob job1(1, UccJob::UCC_JOB_CTX_GLOBAL);
    job1.cleanup();

    UccJob job3(3, UccJob::UCC_JOB_CTX_GLOBAL);
    job3.cleanup();

    UccJob job11(11, UccJob::UCC_JOB_CTX_GLOBAL);
    job11.cleanup();

    UccJob job16(16, UccJob::UCC_JOB_CTX_GLOBAL);
    job16.cleanup();

}
