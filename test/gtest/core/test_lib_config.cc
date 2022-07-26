/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <common/test_ucc.h>

class test_lib_config : public ucc::test {
};

UCC_TEST_F(test_lib_config, read_release)
{
    ucc_lib_config_h cfg;
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));
    ucc_lib_config_release(cfg);
}

UCC_TEST_F(test_lib_config, print)
{
    ucc_lib_config_h cfg;
    unsigned         flags;
    testing::internal::CaptureStdout();
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));
    flags = UCC_CONFIG_PRINT_CONFIG | UCC_CONFIG_PRINT_HEADER |
            UCC_CONFIG_PRINT_DOC | UCC_CONFIG_PRINT_HIDDEN;
    ucc_lib_config_print(cfg, stdout, "TEST_TITLE",
                         (ucc_config_print_flags_t)flags);
    ucc_lib_config_release(cfg);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("# TEST_TITLE"));
    EXPECT_NE(std::string::npos, output.find("UCC_CLS="));
}

UCC_TEST_F(test_lib_config, modify)
{
    ucc_lib_config_h cfg;
    unsigned         flags;
    testing::internal::CaptureStdout();
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));

    /* modify known field to expected value */
    EXPECT_EQ(UCC_OK, ucc_lib_config_modify(cfg, "CLS", "basic"));
    flags = UCC_CONFIG_PRINT_CONFIG;
    ucc_lib_config_print(cfg, stdout, "", (ucc_config_print_flags_t)flags);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(std::string::npos, output.find("UCC_CLS=basic"));

    /* modify known field to unexpected value */
    /* temporarily commented out due to a bug in UCS which results in segv */
    // EXPECT_NE(UCC_OK, ucc_lib_config_modify(cfg, "CLS", "_unknown_value"));

    /* modify uknown field */
    EXPECT_NE(UCC_OK,
              ucc_lib_config_modify(cfg, "_UNKNOWN_FIELD", "_unknown_value"));
    ucc_lib_config_release(cfg);
}
