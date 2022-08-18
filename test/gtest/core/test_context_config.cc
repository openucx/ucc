/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "test_context.h"

test_context_config::test_context_config()
{
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &lib_config));
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    EXPECT_EQ(UCC_OK, ucc_init(&lib_params, lib_config, &lib_h));
    ucc_lib_config_release(lib_config);
}

test_context_config::~test_context_config()
{
    EXPECT_EQ(UCC_OK, ucc_finalize(lib_h));
}

UCC_TEST_F(test_context_config, read_release)
{
    ucc_context_config_h ctx_config;
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));
    ucc_context_config_release(ctx_config);
}

UCC_TEST_F(test_context_config, print)
{
    ucc_context_config_h ctx_config;
    unsigned             flags;
    testing::internal::CaptureStdout();

    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));
    flags = UCC_CONFIG_PRINT_CONFIG | UCC_CONFIG_PRINT_HEADER |
            UCC_CONFIG_PRINT_DOC | UCC_CONFIG_PRINT_HIDDEN;
    ucc_context_config_print(ctx_config, stdout, "TEST_TITLE",
                             (ucc_config_print_flags_t)flags);
    ucc_context_config_release(ctx_config);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("# TEST_TITLE"));
    EXPECT_NE(std::string::npos, output.find("# CL_BASIC"));
}

UCC_TEST_F(test_context_config, modify)
{
    ucc_context_config_h ctx_config;
    std::string          output;
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));

    /* modify uknown field */
    testing::internal::CaptureStdout();
    EXPECT_NE(UCC_OK,
              ucc_context_config_modify(ctx_config, "cl/basic", "_UNKNOWN_FIELD", "_unknown_value"));
    output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("failed to modify"));

    /* modify uknown cl */
    testing::internal::CaptureStdout();
    EXPECT_NE(UCC_OK,
              ucc_context_config_modify(ctx_config, "_unknown_cl", "_UNKNOWN_FIELD", "_unknown_value"));
    output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("invalid component name"));

    /* modify "tl/ucp" */
    EXPECT_EQ(UCC_OK,
              ucc_context_config_modify(ctx_config, "tl/ucp", "NPOLLS", "123"));

    EXPECT_EQ(UCC_OK,
              ucc_context_config_modify(ctx_config, "tl/ucp", "TUNE", "123"));

    ucc_context_config_release(ctx_config);
}

UCC_TEST_F(test_context_config, modify_core)
{
    ucc_context_config_h ctx_config;
    unsigned             flags;
    std::string          output;
    flags = UCC_CONFIG_PRINT_CONFIG;
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));

    /* modify known field to expected value */
    EXPECT_EQ(UCC_OK, ucc_context_config_modify(ctx_config, NULL, "ESTIMATED_NUM_EPS", "12345"));
    testing::internal::CaptureStdout();
    ucc_context_config_print(ctx_config, stdout, "", (ucc_config_print_flags_t)flags);
    output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("UCC_ESTIMATED_NUM_EPS=12345"));

    /* modify uknown field */
    testing::internal::CaptureStdout();
    EXPECT_NE(UCC_OK,
              ucc_context_config_modify(ctx_config, NULL, "_UNKNOWN_FIELD", "_unknown_value"));
    output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("failed to modify"));
    ucc_context_config_release(ctx_config);
}
