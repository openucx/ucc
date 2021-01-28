/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include <common/test.h>

class test_mc : public ucc::test {
};

UCC_TEST_F(test_mc, init_finalize)
{
    EXPECT_EQ(UCC_OK, ucc_constructor());
    EXPECT_EQ(UCC_OK, ucc_mc_init());
    EXPECT_EQ(UCC_OK, ucc_mc_finalize());
}

UCC_TEST_F(test_mc, init_is_required)
{
    ASSERT_EQ(UCC_OK, ucc_constructor());
    EXPECT_NE(UCC_OK, ucc_mc_available(UCC_MEMORY_TYPE_HOST));
    EXPECT_EQ(UCC_OK, ucc_mc_init());
    EXPECT_EQ(UCC_OK, ucc_mc_available(UCC_MEMORY_TYPE_HOST));
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc, can_alloc_and_free_host_mem)
{
    void *ptr = NULL;
    size_t size = 4096;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());
    EXPECT_EQ(UCC_OK, ucc_mc_alloc(&ptr, size, UCC_MEMORY_TYPE_HOST));
    memset(ptr, 0, size);
    EXPECT_EQ(UCC_OK, ucc_mc_free(ptr, UCC_MEMORY_TYPE_HOST));
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc, init_twice)
{
    ucc_lib_config_h cfg;
    ucc_lib_params_t lib_params;
    ucc_lib_h lib1, lib2;
    ucc_mc_base_t *mc;

    ASSERT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &cfg));
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    ASSERT_EQ(UCC_OK, ucc_init(&lib_params, cfg, &lib1));
    mc = ucc_derived_of(ucc_global_config.mc_framework.components[0],
                        ucc_mc_base_t);
    EXPECT_EQ(1, mc->ref_cnt);
    EXPECT_EQ(UCC_OK, ucc_init(&lib_params, cfg, &lib2));
    EXPECT_EQ(2, mc->ref_cnt);
    EXPECT_EQ(UCC_OK, ucc_finalize(lib1));
    EXPECT_EQ(1, mc->ref_cnt);
    EXPECT_EQ(UCC_OK, ucc_finalize(lib2));
    EXPECT_EQ(0, mc->ref_cnt);

    ucc_lib_config_release(cfg);
}
