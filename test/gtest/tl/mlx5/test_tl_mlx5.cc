/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_tl_mlx5.h"

test_tl_mlx5::test_tl_mlx5()
{
    tl_mlx5_so_handle     = NULL;
    ctx                   = NULL;
    pd                    = NULL;
    cq                    = NULL;
}

void test_tl_mlx5::SetUp()
{
    char *       devname = NULL;
    ucc_status_t status;

    ASSERT_EQ(UCC_OK, ucc_constructor());
    ucc_strncpy_safe(lib.log_component.name, "GTEST_MLX5",
                     sizeof(lib.log_component.name));
    lib.log_component.log_level = UCC_LOG_LEVEL_ERROR;

    std::string path =
        std::string(ucc_global_config.component_path) + "/libucc_tl_mlx5.so";
    tl_mlx5_so_handle = dlopen(path.c_str(), RTLD_NOW);
    if (!tl_mlx5_so_handle) {
        GTEST_SKIP() << "cannot open ucc_tl_mlx5 library" ;
    }

    create_ibv_ctx = (ucc_tl_mlx5_create_ibv_ctx_fn_t)dlsym(
        tl_mlx5_so_handle, "ucc_tl_mlx5_create_ibv_ctx");
    ASSERT_EQ(nullptr, dlerror());

    get_active_port = (ucc_tl_mlx5_get_active_port_fn_t)dlsym(
        tl_mlx5_so_handle, "ucc_tl_mlx5_get_active_port");
    ASSERT_EQ(nullptr, dlerror());

    status = create_ibv_ctx(&devname, &ctx, &lib);
    if (UCC_OK != status) {
        GTEST_SKIP() << "no ib devices";
    }
    port = get_active_port(ctx);
    ASSERT_GE(port, 0);

    ASSERT_EQ(ibv_query_port(ctx, port, &port_attr), 0);

    pd = ibv_alloc_pd(ctx);
    ASSERT_NE(nullptr, pd);

    cq = ibv_create_cq(ctx, 8, NULL, NULL, 0);
    ASSERT_NE(nullptr, cq);
}

test_tl_mlx5::~test_tl_mlx5()
{
    if (cq) {
        ibv_destroy_cq(cq);
    }
    if (pd) {
        ibv_dealloc_pd(pd);
    }
    if (ctx) {
        ibv_close_device(ctx);
    }
    if (tl_mlx5_so_handle) {
        dlclose(tl_mlx5_so_handle);
    }
}
