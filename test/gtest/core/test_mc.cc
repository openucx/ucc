/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include <components/mc/ucc_mc.h>
#include <pthread.h>
}
#include <common/test.h>
#include <vector>

void *mt_ucc_mc_cpu_allocs(void *args)
{
    // Final size will be:
    // size * (quantifier^(num_of_allocs/2))
    // and should be larger than mpool buffer size which is 1MB by default,
    // to assure testing both fast and slow ucc_mc_alloc path.
    // if num_of_allocs is changed, change quantifier accordingly.
    size_t                                size          = 4;
    int                                   quantifier    = 2;
    int                                   num_of_allocs = 40;
    std::vector<ucc_mc_buffer_header_t *> headers;
    std::vector<void *>                   pointers;
    headers.resize(num_of_allocs);
    pointers.resize(num_of_allocs);

    for (int i = 0; i < num_of_allocs; i++) {
        EXPECT_EQ(UCC_OK,
                  ucc_mc_alloc(&headers[i], size, UCC_MEMORY_TYPE_HOST));
        pointers[i] = headers[i]->addr;
        memset(pointers[i], 0, size);
        EXPECT_EQ(UCC_OK, ucc_mc_free(headers[i]));
        if (i % 2) {
            size *= quantifier;
        }
    }
    return 0;
}

class test_mc : public ucc::test {
};

UCC_TEST_F(test_mc, init_finalize)
{
    EXPECT_EQ(UCC_OK, ucc_constructor());
    ucc_mc_params_t mc_params = {
        .thread_mode = UCC_THREAD_SINGLE,
    };
    EXPECT_EQ(UCC_OK, ucc_mc_init(&mc_params));
    EXPECT_EQ(UCC_OK, ucc_mc_finalize());
}

UCC_TEST_F(test_mc, init_is_required)
{
    ASSERT_EQ(UCC_OK, ucc_constructor());
    EXPECT_NE(UCC_OK, ucc_mc_available(UCC_MEMORY_TYPE_HOST));
    ucc_mc_params_t mc_params = {
        .thread_mode = UCC_THREAD_SINGLE,
    };
    EXPECT_EQ(UCC_OK, ucc_mc_init(&mc_params));
    EXPECT_EQ(UCC_OK, ucc_mc_available(UCC_MEMORY_TYPE_HOST));
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc, can_alloc_and_free_host_mem)
{
    // mpool will be used only if size is smaller than UCC_MC_CPU_ELEM_SIZE, which by default set to 1MB and is configurable at runtime.
    size_t                  size = 4096;
    ucc_mc_buffer_header_t *h;
    void *ptr = NULL;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ucc_mc_params_t mc_params = {
        .thread_mode = UCC_THREAD_SINGLE,
    };
    ASSERT_EQ(UCC_OK, ucc_mc_init(&mc_params));
    EXPECT_EQ(UCC_OK, ucc_mc_alloc(&h, size, UCC_MEMORY_TYPE_HOST));
    ptr = h->addr;
    memset(ptr, 0, size);
    EXPECT_EQ(UCC_OK, ucc_mc_free(h));
    ucc_mc_finalize();
}

// Disabled because can't reinit mc with different thread mode
UCC_TEST_F(test_mc, DISABLED_can_alloc_and_free_host_mem_mt)
{
    // mpool will be used only if size is smaller than UCC_MC_CPU_ELEM_SIZE, which by default set to 1MB and is configurable at runtime.
    int                    num_of_threads = 10;
    std::vector<pthread_t> threads;
    threads.resize(num_of_threads);

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ucc_mc_params_t mc_params = {
        .thread_mode = UCC_THREAD_MULTIPLE,
    };
    ASSERT_EQ(UCC_OK, ucc_mc_init(&mc_params));
    for (int i = 0; i < num_of_threads; i++) {
        pthread_create(&threads[i], NULL, &mt_ucc_mc_cpu_allocs, NULL);
    }
    for (int i = 0; i < num_of_threads; i++) {
        pthread_join(threads[i], NULL);
    }
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
