/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>
class test_lib : public ucc::test {};

UCC_TEST_F(test_lib, init_finalize)
{
    ucc_lib_config_h cfg;
    ucc_lib_params_t lib_params;
    ucc_lib_h lib;
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));
    lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    EXPECT_EQ(UCC_OK, ucc_init(&lib_params, cfg, &lib));
    ucc_lib_config_release(cfg);
    EXPECT_EQ(UCC_OK, ucc_finalize(lib));
}
