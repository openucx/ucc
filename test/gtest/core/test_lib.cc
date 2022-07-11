/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include <common/test_ucc.h>
#include <algorithm>
#include <random>

class test_lib : public ucc::test {
};

UCC_TEST_F(test_lib, init_finalize)
{
    ucc_lib_config_h cfg;
    ucc_lib_params_t lib_params;
    ucc_lib_h        lib;
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    EXPECT_EQ(UCC_OK, ucc_init(&lib_params, cfg, &lib));
    ucc_lib_config_release(cfg);
    EXPECT_EQ(UCC_OK, ucc_finalize(lib));
}

UCC_TEST_F(test_lib, init_multiple)
{
    const int              n_libs = 8;
    ucc_lib_config_h       cfg;
    ucc_lib_params_t       lib_params;
    ucc_lib_h              lib_h;
    std::vector<ucc_lib_h> libs;
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    for (int i = 0; i < n_libs; i++) {
        EXPECT_EQ(UCC_OK, ucc_init(&lib_params, cfg, &lib_h));
        libs.push_back(lib_h);
    }
    ucc_lib_config_release(cfg);
    std::shuffle(libs.begin(), libs.end(), std::default_random_engine());

    for (auto lib_h : libs) {
        EXPECT_EQ(UCC_OK, ucc_finalize(lib_h));
    }
}
