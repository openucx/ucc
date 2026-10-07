/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_mem_map.h"
#include <cstring>
#include <random>
extern "C" {
#include "core/ucc_context.h"
}

test_mem_map::test_mem_map() : ctx_h(nullptr), ctx_config(nullptr)
{
    memset(&ctx_params, 0, sizeof(ctx_params));
}

test_mem_map::~test_mem_map()
{
}

void test_mem_map::SetUp()
{
    test_context_config::SetUp();
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));

    ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE;
    ctx_params.type = UCC_CONTEXT_EXCLUSIVE;
    EXPECT_EQ(UCC_OK, ucc_context_create(lib_h, &ctx_params, ctx_config,
                                         &ctx_h));
}

void test_mem_map::TearDown()
{
    if (ctx_h) {
        EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
    }
    if (ctx_config) {
        ucc_context_config_release(ctx_config);
    }
    test_context_config::TearDown();
}

test_mem_map_export::test_mem_map_export() : test_buffer(nullptr), buffer_size(0)
{
    memset(&map_params, 0, sizeof(map_params));
    memset(&segment, 0, sizeof(segment));
}

test_mem_map_export::~test_mem_map_export()
{
}

void test_mem_map_export::SetUp()
{
    test_mem_map::SetUp();

    /* Allocate test buffer */
    buffer_size = 1024 * 1024; /* 1MB */
    test_buffer = malloc(buffer_size);
    ASSERT_NE(nullptr, test_buffer);

    /* Initialize buffer with test data */
    memset(test_buffer, 0xAA, buffer_size);

    /* Set up memory map parameters */
    segment.address = test_buffer;
    segment.len     = buffer_size;

    map_params.segments   = &segment;
    map_params.n_segments = 1;
}

void test_mem_map_export::TearDown()
{
    if (test_buffer) {
        free(test_buffer);
        test_buffer = nullptr;
    }
    test_mem_map::TearDown();
}

test_mem_map_import::test_mem_map_import()
    : test_buffer(nullptr), buffer_size(0), memh(nullptr), memh_size(0)
{
}

test_mem_map_import::~test_mem_map_import()
{
}

void test_mem_map_import::SetUp()
{
    test_mem_map::SetUp();

    /* Allocate test buffer */
    buffer_size = 1024 * 1024; /* 1MB */
    test_buffer = malloc(buffer_size);
    ASSERT_NE(nullptr, test_buffer);

    /* Initialize buffer with test data */
    memset(test_buffer, 0xBB, buffer_size);
}

void test_mem_map_import::TearDown()
{
    if (test_buffer) {
        free(test_buffer);
        test_buffer = nullptr;
    }
    test_mem_map::TearDown();
}

/* Test basic memory map export functionality */
UCC_TEST_F(test_mem_map_export, basic_export)
{
    ucc_mem_map_mem_h memh = nullptr;
    size_t            memh_size = 0;

    EXPECT_EQ(UCC_OK, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                   &map_params, &memh_size, &memh));
    EXPECT_NE(nullptr, memh);
    EXPECT_GT(memh_size, 0);

    /* Test unmap */
    EXPECT_EQ(UCC_OK, ucc_mem_unmap(&memh));
    /* Note: ucc_mem_unmap doesn't set memh to nullptr, it only frees the memory */
}

/* Test memory map export with different buffer sizes */
UCC_TEST_F(test_mem_map_export, different_sizes)
{
    std::vector<size_t> sizes = {1024, 4096, 65536, 1024 * 1024};

    for (auto size : sizes) {
        /* Reallocate buffer with new size */
        if (test_buffer) {
            free(test_buffer);
        }
        test_buffer = malloc(size);
        ASSERT_NE(nullptr, test_buffer);
        memset(test_buffer, 0xCC, size);

        segment.address = test_buffer;
        segment.len     = size;

        ucc_mem_map_mem_h memh = nullptr;
        size_t             memh_size = 0;

        EXPECT_EQ(UCC_OK, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                       &map_params, &memh_size, &memh));
        EXPECT_NE(nullptr, memh);
        EXPECT_GT(memh_size, 0);

        EXPECT_EQ(UCC_OK, ucc_mem_unmap(&memh));
    }
}

/* Test memory map export with multiple segments (should fail as UCC only supports one segment) */
UCC_TEST_F(test_mem_map_export, multiple_segments)
{
    ucc_mem_map_mem_h    memh      = nullptr;
    size_t               memh_size = 0;
    ucc_mem_map_t        segments[2];
    ucc_mem_map_params_t multi_params;

    /* Create two segments */
    segments[0].address     = test_buffer;
    segments[0].len         = buffer_size / 2;
    segments[1].address     = (char *)test_buffer + buffer_size / 2;
    segments[1].len         = buffer_size / 2;
    multi_params.segments   = segments;
    multi_params.n_segments = 2;

    /* This should fail as UCC only supports one segment per call */
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                                   &multi_params, &memh_size,
                                                   &memh));
    EXPECT_EQ(nullptr, memh);
}

/* Test memory map export with invalid parameters */
UCC_TEST_F(test_mem_map_export, invalid_params)
{
    ucc_mem_map_mem_h memh = nullptr;
    size_t            memh_size = 0;

    /* Test with NULL params */
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                                   nullptr, &memh_size, &memh));

    /* Test with invalid mode */
    ucc_mem_map_mode_t invalid_mode = UCC_MEM_MAP_MODE_LAST;
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, ucc_mem_map(ctx_h, invalid_mode,
                                                   &map_params, &memh_size,
                                                   &memh));
}

/* Test memory map export with zero length buffer */
UCC_TEST_F(test_mem_map_export, zero_length)
{
    ucc_mem_map_mem_h  memh      = nullptr;
    size_t             memh_size = 0;

    segment.len = 0;
    /* This might succeed or fail depending on implementation */
    ucc_status_t status = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                       &map_params, &memh_size, &memh);
    if (status == UCC_OK) {
        EXPECT_NE(nullptr, memh);
        EXPECT_EQ(UCC_OK, ucc_mem_unmap(&memh));
    }
}

/* Test memory map import functionality */
UCC_TEST_F(test_mem_map_import, basic_import)
{
    ucc_mem_map_mem_h    export_memh      = nullptr;
    size_t               export_memh_size = 0;
    ucc_mem_map_mem_h    import_memh      = nullptr;
    size_t               import_memh_size = 0;
    ucc_mem_map_t        export_segment;
    ucc_mem_map_params_t export_params;
    ucc_status_t         export_status;
    ucc_status_t         import_status;

    export_segment.address   = test_buffer;
    export_segment.len       = buffer_size;
    export_params.segments   = &export_segment;
    export_params.n_segments = 1;

    /* Export the memory handle */
    export_status = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                &export_params, &export_memh_size,
                                &export_memh);

    if (export_status != UCC_OK) {
        /* If export fails, skip the test */
        GTEST_SKIP() << "Export failed, skipping import test";
        return;
    }

    EXPECT_NE(nullptr, export_memh);
    EXPECT_GT(export_memh_size, 0);

    /* For import, we need to create a new memory handle with the exported data */
    /* The import function expects the memh to be pre-allocated with the exported data */
    import_memh = (ucc_mem_map_mem_h)malloc(export_memh_size);
    ASSERT_NE(nullptr, import_memh);
    memcpy(import_memh, export_memh, export_memh_size);

    import_status = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                &export_params, &import_memh_size,
                                &import_memh);

    if (import_status == UCC_OK) {
        EXPECT_NE(nullptr, import_memh);

        /* Cleanup import */
        ucc_mem_unmap(&import_memh);
    } else {
        /* Import might not be supported, which is acceptable */
        EXPECT_TRUE(import_status == UCC_ERR_NOT_SUPPORTED ||
                    import_status == UCC_ERR_NOT_IMPLEMENTED);

        /* Clean up the allocated memory if import failed */
        free(import_memh);
    }

    /* Cleanup export */
    ucc_mem_unmap(&export_memh);
}

/* Test memory map import with different buffer sizes */
UCC_TEST_F(test_mem_map_import, import_different_sizes)
{
    std::vector<size_t>  sizes            = {1024, 4096, 65536, 1024 * 1024};
    ucc_mem_map_mem_h    export_memh      = nullptr;
    size_t               export_memh_size = 0;
    ucc_mem_map_t        export_segment;
    ucc_mem_map_params_t export_params;
    ucc_mem_map_mem_h    import_memh;
    size_t               import_memh_size;
    ucc_status_t         import_status;

    for (auto size : sizes) {
        /* Reallocate buffer with new size */
        if (test_buffer) {
            free(test_buffer);
        }
        test_buffer = malloc(size);
        ASSERT_NE(nullptr, test_buffer);
        memset(test_buffer, 0xDD, size);

        export_segment.address   = test_buffer;
        export_segment.len       = size;
        export_params.segments   = &export_segment;
        export_params.n_segments = 1;

        /* Export the memory handle */

        ucc_status_t export_status = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                                 &export_params, &export_memh_size,
                                                 &export_memh);

        if (export_status != UCC_OK) {
            continue; /* Skip this size if export fails */
        }

        EXPECT_NE(nullptr, export_memh);
        EXPECT_GT(export_memh_size, 0);

        /* Test import */
        import_memh = (ucc_mem_map_mem_h)malloc(export_memh_size);
        ASSERT_NE(nullptr, import_memh);
        memcpy(import_memh, export_memh, export_memh_size);

        import_status = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                   &export_params, &import_memh_size,
                                   &import_memh);
        if (import_status == UCC_OK) {
            EXPECT_NE(nullptr, import_memh);
            ucc_mem_unmap(&import_memh);
        } else {
            /* Import might not be supported for all sizes */
            EXPECT_TRUE(import_status == UCC_ERR_NOT_SUPPORTED ||
                        import_status == UCC_ERR_NOT_IMPLEMENTED);

            /* Clean up the allocated memory if import failed */
            free(import_memh);
        }
        /* Cleanup export */
        ucc_mem_unmap(&export_memh);
    }
}

/* Test memory map import with invalid parameters */
UCC_TEST_F(test_mem_map_import, import_invalid_params)
{
    /* Test import with NULL params */
    ucc_mem_map_mem_h    memh      = nullptr;
    size_t               memh_size = 0;
    ucc_mem_map_params_t params;
    ucc_mem_map_t        segment;

    EXPECT_EQ(UCC_ERR_INVALID_PARAM, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                                 nullptr, &memh_size, &memh));

    /* Test import with NULL memh */
    segment.address   = test_buffer;
    segment.len       = buffer_size;
    params.segments   = &segment;
    params.n_segments = 1;

    EXPECT_EQ(UCC_ERR_INVALID_PARAM, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                                &params, &memh_size, nullptr));
}

/* Test memory map unmap with NULL handle */
UCC_TEST_F(test_mem_map_export, unmap_null)
{
    ucc_mem_map_mem_h memh = nullptr;

    /* Should handle NULL gracefully */
    EXPECT_EQ(UCC_ERR_INVALID_PARAM, ucc_mem_unmap(&memh));
}

/* Test memory map with different modes */
UCC_TEST_F(test_mem_map_export, different_modes)
{
    ucc_mem_map_mem_h memh1     = nullptr;
    size_t            memh_size = 0;

    /* Test EXPORT mode */
    EXPECT_EQ(UCC_OK, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                   &map_params, &memh_size, &memh1));
    EXPECT_NE(nullptr, memh1);
    EXPECT_EQ(UCC_OK, ucc_mem_unmap(&memh1));
}

/* Test memory map stress test */
UCC_TEST_F(test_mem_map_export, stress_test)
{
    const int         num_iterations = 100;
    ucc_mem_map_mem_h memh;
    size_t            memh_size;

    for (int i = 0; i < num_iterations; i++) {
        memh      = nullptr;
        memh_size = 0;

        /* Fill buffer with iteration-specific pattern */
        memset(test_buffer, i % 256, buffer_size);

        EXPECT_EQ(UCC_OK, ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT,
                                     &map_params, &memh_size, &memh));
        EXPECT_NE(nullptr, memh);

        EXPECT_EQ(UCC_OK, ucc_mem_unmap(&memh));
    }
}

/* Test that unmap-export before unmap-import is safe (reverse lifecycle order).
 * basic_import only exercises unmap-import first; this covers the other path
 * through ucc_tl_nccl_team_deregister_memhs and ucc_tl_nccl_mem_unmap. */
UCC_TEST_F(test_mem_map_import, unmap_export_before_import)
{
    ucc_mem_map_t        seg      = {test_buffer, buffer_size};
    ucc_mem_map_params_t params   = {&seg, 1};
    ucc_mem_map_mem_h    exp      = nullptr;
    ucc_mem_map_mem_h    imp      = nullptr;
    size_t               exp_size = 0;
    size_t               imp_size = 0;

    if (ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT, &params,
                    &exp_size, &exp) != UCC_OK) {
        GTEST_SKIP() << "export not supported";
    }
    imp = malloc(exp_size);
    ASSERT_NE(nullptr, imp);
    memcpy(imp, exp, exp_size);

    ucc_status_t st = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                  &params, &imp_size, &imp);
    if (st != UCC_OK) {
        free(imp);
        ucc_mem_unmap(&exp);
        GTEST_SKIP() << "import not supported";
    }

    /* Destroy export first, then the import handle. */
    EXPECT_EQ(UCC_OK, ucc_mem_unmap(&exp));
    EXPECT_EQ(UCC_OK, ucc_mem_unmap(&imp));
}

/* Test that import does not overwrite memh->address/len with the local VA.
 * basic_import passes the same params to both export and import so it cannot
 * distinguish a bug where the remote address is silently replaced.  This test
 * uses a separate local buffer so the two addresses are distinct. */
UCC_TEST_F(test_mem_map_import, import_preserves_remote_address)
{
    ucc_mem_map_t        seg      = {test_buffer, buffer_size};
    ucc_mem_map_params_t params   = {&seg, 1};
    ucc_mem_map_mem_h    exp      = nullptr;
    ucc_mem_map_mem_h    imp      = nullptr;
    size_t               exp_size = 0;
    size_t               imp_size = 0;
    void                *local_buf;

    if (ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT, &params,
                    &exp_size, &exp) != UCC_OK) {
        GTEST_SKIP() << "export not supported";
    }
    imp = malloc(exp_size);
    ASSERT_NE(nullptr, imp);
    memcpy(imp, exp, exp_size);

    /* Use a distinct local buffer so remote VA != local VA. */
    local_buf = malloc(buffer_size);
    ASSERT_NE(nullptr, local_buf);
    ucc_mem_map_t        local_seg    = {local_buf, buffer_size};
    ucc_mem_map_params_t local_params = {&local_seg, 1};

    ucc_status_t st = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                  &local_params, &imp_size, &imp);
    if (st == UCC_OK) {
        ucc_mem_map_memh_t *ih = (ucc_mem_map_memh_t *)imp;

        /* address/len must still carry the remote (export) values. */
        EXPECT_EQ(test_buffer, ih->address)
            << "import must not overwrite memh->address with local VA";
        EXPECT_EQ(buffer_size, ih->len)
            << "import must not overwrite memh->len";

        /* Local values are in the dedicated import fields. */
        EXPECT_EQ(local_buf,   ih->local_address);
        EXPECT_EQ(buffer_size, ih->local_len);

        EXPECT_EQ(UCC_OK, ucc_mem_unmap(&imp));
    } else {
        free(imp);
    }
    free(local_buf);
    EXPECT_EQ(UCC_OK, ucc_mem_unmap(&exp));
}

/* Test repeated export+import+unmap cycles to catch cumulative leaks or
 * corruption in the bookkeeping arrays.  Alternates unmap order each cycle. */
UCC_TEST_F(test_mem_map_import, repeated_export_import_cycles)
{
    const int            num_cycles = 20;
    ucc_mem_map_t        seg        = {test_buffer, buffer_size};
    ucc_mem_map_params_t params     = {&seg, 1};
    bool                 skipped    = false;

    for (int c = 0; c < num_cycles; c++) {
        ucc_mem_map_mem_h exp      = nullptr;
        ucc_mem_map_mem_h imp      = nullptr;
        size_t            exp_size = 0;
        size_t            imp_size = 0;

        if (ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_EXPORT, &params,
                        &exp_size, &exp) != UCC_OK) {
            skipped = true;
            break;
        }
        imp = malloc(exp_size);
        ASSERT_NE(nullptr, imp);
        memcpy(imp, exp, exp_size);

        ucc_status_t st = ucc_mem_map(ctx_h, UCC_MEM_MAP_MODE_IMPORT,
                                      &params, &imp_size, &imp);
        if (st != UCC_OK) {
            free(imp);
            ucc_mem_unmap(&exp);
            skipped = true;
            break;
        }

        if (c % 2 == 0) {
            EXPECT_EQ(UCC_OK, ucc_mem_unmap(&imp));
            EXPECT_EQ(UCC_OK, ucc_mem_unmap(&exp));
        } else {
            EXPECT_EQ(UCC_OK, ucc_mem_unmap(&exp));
            EXPECT_EQ(UCC_OK, ucc_mem_unmap(&imp));
        }
    }

    if (skipped) {
        GTEST_SKIP() << "export/import not supported";
    }
}
