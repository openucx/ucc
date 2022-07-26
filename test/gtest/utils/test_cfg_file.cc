/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "utils/ucc_parser.h"
#include "core/ucc_global_opts.h"
}
#include <common/test.h>

typedef struct ucc_gtest_config_base {
    int foo;
} ucc_gtest_config_base_t;

typedef struct ucc_gtest_config {
    ucc_gtest_config_base_t super;
    int                     bar;
    int                     boo;
} ucc_gtest_config_t;


static ucc_config_field_t ucc_gtest_config_table_base[] = {
    {"FOO", "1", "gtest base config variable",
     ucc_offsetof(ucc_gtest_config_base_t, foo),
     UCC_CONFIG_TYPE_INT},

    {NULL}
};

static ucc_config_field_t ucc_gtest_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_gtest_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_gtest_config_table_base)},

    {"BAR", "1", "gtest config variable",
     ucc_offsetof(ucc_gtest_config_t, bar),
     UCC_CONFIG_TYPE_INT},

    {"BOO", "1", "gtest config variable",
     ucc_offsetof(ucc_gtest_config_t, boo),
     UCC_CONFIG_TYPE_INT},

    {NULL}
};

class test_cfg_file : public ucc::test {
public:
    ucc_file_config_t  *file_cfg;
    ucc_gtest_config_t  cfg;
    std::string         test_dir;
    test_cfg_file() {
        file_cfg = NULL;
        test_dir = std::string(GTEST_UCC_TOP_SRCDIR) +
            "/test/gtest/utils/";
    }
    ~test_cfg_file() {
        if (file_cfg) {
            ucc_release_file_config(file_cfg);
        }
        ucc_config_parser_release_opts(&cfg, ucc_gtest_config_table);
    }
    void init_cfg() {
        ucc_status_t status;
        std::swap(ucc_global_config.file_cfg, file_cfg);
        status = ucc_config_parser_fill_opts(&cfg, ucc_gtest_config_table,
                                             "GTEST_UCC_", "CFG_", 1);
        std::swap(ucc_global_config.file_cfg, file_cfg);
        EXPECT_EQ(UCC_OK, status);
    };
};

UCC_TEST_F(test_cfg_file, parse_existing) {
    std::string filename = test_dir + "ucc_test.conf";

    EXPECT_EQ(UCC_OK, ucc_parse_file_config(filename.c_str(), &file_cfg));
}

UCC_TEST_F(test_cfg_file, parse_non_existing) {
    std::string filename = test_dir + "ucc_test_nonexisting.conf";

    EXPECT_EQ(UCC_ERR_NOT_FOUND, ucc_parse_file_config(filename.c_str(),
                                                       &file_cfg));
}

/* Checks options are applied from cfg file */
UCC_TEST_F(test_cfg_file, opts_applied) {
    std::string filename = test_dir + "ucc_test.conf";

    EXPECT_EQ(UCC_OK, ucc_parse_file_config(filename.c_str(), &file_cfg));
    init_cfg();
    EXPECT_EQ(10, cfg.super.foo);
    EXPECT_EQ(20, cfg.bar);
    EXPECT_EQ(1,  cfg.boo);
}

/* Checks that options set via env var have preference
   over cfg file */
UCC_TEST_F(test_cfg_file, env_preference) {
    std::string filename = test_dir + "ucc_test.conf";

    setenv("GTEST_UCC_CFG_BAR", "123", 1);
    EXPECT_EQ(UCC_OK, ucc_parse_file_config(filename.c_str(), &file_cfg));
    init_cfg();
    unsetenv("GTEST_UCC_CFG_BAR");
    EXPECT_EQ(10, cfg.super.foo);
    /* Expected value is 123 from env rather than 20 from file */
    EXPECT_EQ(123, cfg.bar);
    EXPECT_EQ(1,  cfg.boo);
}
