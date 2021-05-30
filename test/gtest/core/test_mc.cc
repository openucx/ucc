/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
#include <pthread.h>
}
#include <common/test.h>
#include <vector>

void *mt_ucc_mc_cpu_calls(void * args)
{
	size_t *size = (size_t *) args;
	int num_of_alloc_calls = 50;
	std::vector<ucc_mc_buffer_header_t *> headers;
	std::vector<void *> pointers;

	headers.resize(num_of_alloc_calls);
	pointers.resize(num_of_alloc_calls);

	for (int i = 0; i < num_of_alloc_calls; i++){
		pointers[i] = NULL;
		EXPECT_EQ(UCC_OK, ucc_mc_alloc(&headers[i], *size, UCC_MEMORY_TYPE_HOST));
		pointers[i] = headers[i]->addr;
		memset(pointers[i], 0, *size);
		EXPECT_EQ(UCC_OK, ucc_mc_free(headers[i], UCC_MEMORY_TYPE_HOST));
	}

	return 0;
}

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
	// mpool will be used only if size is smaller than UCC_MC_CPU_ELEM_SIZE, which by default set to 1024 and is configurable at runtime.
	size_t size = 4096;
    ucc_mc_buffer_header_t *h    = NULL;
    void *ptr = NULL;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());
    EXPECT_EQ(UCC_OK, ucc_mc_alloc(&h, size, UCC_MEMORY_TYPE_HOST));
    ptr = h->addr;
    memset(ptr, 0, size);
    EXPECT_EQ(UCC_OK, ucc_mc_free(h, UCC_MEMORY_TYPE_HOST));
    ucc_mc_finalize();
}

UCC_TEST_F(test_mc, can_alloc_and_free_host_mem_mt)
{
	// mpool will be used only if size is smaller than UCC_MC_CPU_ELEM_SIZE, which by default set to 1024 and is configurable at runtime.
	size_t size = 1024;
    int num_of_threads = 20;

    std::vector<pthread_t> threads;
    threads.resize(num_of_threads);
    ASSERT_EQ(UCC_OK, ucc_constructor());
    ASSERT_EQ(UCC_OK, ucc_mc_init());
    for (int i = 0; i < num_of_threads; i++) {
    	pthread_create(&threads[i], NULL, &mt_ucc_mc_cpu_calls, (void *)&size);
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
