/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "test_context.h"
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
    ctx_params.ctx_type = UCC_CONTEXT_EXCLUSIVE;
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
    ctx_params.ctx_type = UCC_CONTEXT_EXCLUSIVE;
    for (int i = 0; i < n_ctxs; i++) {
        EXPECT_EQ(UCC_OK, ucc_context_create(lib_h, &ctx_params, ctx_config, &ctx_h));
        ctxs.push_back(ctx_h);
    }

    std::shuffle(ctxs.begin(), ctxs.end(), std::default_random_engine());
    for (auto ctx_h : ctxs) {
        EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
    }
}
