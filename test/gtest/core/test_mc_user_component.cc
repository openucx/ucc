/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include <components/mc/ucc_mc.h>
#include <components/mc/ucc_mc_user_component.h>
#include <utils/ucc_component.h>
#include <core/ucc_global_opts.h>
}
#include <common/test.h>
#include <cstdlib>
#include <cstring>

static int               g_finalize_called = 0;
static ucc_memory_type_t g_assigned_type_a = UCC_MEMORY_TYPE_LAST;

static ucc_status_t mock_init(const ucc_mc_params_t *) { return UCC_OK; }

static ucc_status_t mock_get_attr(ucc_mc_attr_t *attr)
{
    attr->thread_mode = UCC_THREAD_SINGLE;
    return UCC_OK;
}

static ucc_status_t mock_finalize_fn()
{
    g_finalize_called++;
    return UCC_OK;
}

static ucc_status_t mock_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                   size_t size, ucc_memory_type_t mt)
{
    ucc_mc_buffer_header_t *h =
        (ucc_mc_buffer_header_t *)malloc(sizeof(*h) + size);
    if (!h) {
        return UCC_ERR_NO_MEMORY;
    }
    h->mt        = mt;
    h->from_pool = 0;
    h->addr      = (char *)h + sizeof(*h);
    *h_ptr       = h;
    return UCC_OK;
}

static ucc_status_t mock_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    free(h_ptr);
    return UCC_OK;
}

static ucc_status_t mock_mem_query_a(const void *, ucc_mem_attr_t *mem_attr)
{
    mem_attr->mem_type = g_assigned_type_a;
    return UCC_OK;
}

class test_mc_user_component : public ucc::test {
protected:
    ucc_mc_base_t mc_a, mc_b;

    void make_mock(ucc_mc_base_t *mc, const char *name,
                   ucc_status_t (*mem_query)(const void *, ucc_mem_attr_t *))
    {
        memset(mc, 0, sizeof(*mc));
        mc->super.name    = name;
        mc->ee_type       = UCC_EE_CPU_THREAD;
        mc->type          = UCC_MEMORY_TYPE_LAST;
        mc->ref_cnt       = 0;
        mc->init          = mock_init;
        mc->get_attr      = mock_get_attr;
        mc->finalize      = mock_finalize_fn;
        mc->ops.mem_alloc = mock_mem_alloc;
        mc->ops.mem_free  = mock_mem_free;
        mc->ops.mem_query = mem_query;
    }

    virtual void SetUp() override
    {
        ucc::test::SetUp();
        ucc_constructor();
        ucc_mc_params_t mc_params = {.thread_mode = UCC_THREAD_SINGLE};
        ucc_mc_init(&mc_params);

        g_finalize_called = 0;
        g_assigned_type_a = UCC_MEMORY_TYPE_LAST;

        make_mock(&mc_a, "mock_a", mock_mem_query_a);
        make_mock(&mc_b, "mock_b", NULL);
    }

    virtual void TearDown() override
    {
        ucc_mc_finalize();
        ucc::test::TearDown();
    }
};

/* Assigned type is beyond UCC_MEMORY_TYPE_LAST and is_user_component returns true */
UCC_TEST_F(test_mc_user_component, register_assigns_type_beyond_last)
{
    ucc_memory_type_t mt;

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt));
    EXPECT_GT(mt, UCC_MEMORY_TYPE_LAST);
    EXPECT_EQ(1, ucc_mc_is_user_component(mt));
    EXPECT_EQ(0, ucc_mc_is_user_component(UCC_MEMORY_TYPE_HOST));
}

/* Registering the same component name twice is rejected */
UCC_TEST_F(test_mc_user_component, register_duplicate_name_fails)
{
    ucc_memory_type_t mt;

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt));
    EXPECT_EQ(UCC_ERR_NO_RESOURCE, ucc_mc_user_component_register(&mc_a, &mt));
}

/* Component without mem_alloc is rejected */
UCC_TEST_F(test_mc_user_component, register_missing_ops_fails)
{
    ucc_memory_type_t mt;

    mc_a.ops.mem_alloc = NULL;
    EXPECT_NE(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt));
}

/* ucc_mc_available reflects registration and unregistration */
UCC_TEST_F(test_mc_user_component, available_tracks_registration)
{
    ucc_memory_type_t mt;

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt));
    EXPECT_EQ(UCC_OK, ucc_mc_available(mt));

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_unregister(mt));
    EXPECT_NE(UCC_OK, ucc_mc_available(mt));
}

/* Unregistering an unknown type returns not-found */
UCC_TEST_F(test_mc_user_component, unregister_unknown_type_fails)
{
    /* Use a type beyond LAST that was never registered. Cast via int to avoid
     * clang-analyzer EnumCastOutOfRange on the intentionally out-of-range value. */
    int               bogus_val = (int)UCC_MEMORY_TYPE_LAST + 99;
    ucc_memory_type_t bogus;
    memcpy(&bogus, &bogus_val, sizeof(bogus));
    EXPECT_EQ(UCC_ERR_NOT_FOUND, ucc_mc_user_component_unregister(bogus));
}

/* Unregistering a builtin type is rejected */
UCC_TEST_F(test_mc_user_component, unregister_builtin_type_fails)
{
    EXPECT_EQ(UCC_ERR_INVALID_PARAM,
              ucc_mc_user_component_unregister(UCC_MEMORY_TYPE_HOST));
}

/* Two components receive distinct, monotonically increasing types */
UCC_TEST_F(test_mc_user_component, multiple_components_get_distinct_types)
{
    ucc_memory_type_t mt_a, mt_b;

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt_a));
    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_b, &mt_b));
    EXPECT_NE(mt_a, mt_b);
    EXPECT_GT(mt_a, UCC_MEMORY_TYPE_LAST);
    EXPECT_GT(mt_b, UCC_MEMORY_TYPE_LAST);
}

/* alloc and free round-trip through user component ops */
UCC_TEST_F(test_mc_user_component, alloc_free_roundtrip)
{
    ucc_memory_type_t       mt;
    ucc_mc_buffer_header_t *h;

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt));
    mc_a.type = mt;

    ASSERT_EQ(UCC_OK, ucc_mc_alloc(&h, 256, mt));
    EXPECT_NE(nullptr, h);
    EXPECT_NE(nullptr, h->addr);
    EXPECT_EQ(mt, h->mt);
    memset(h->addr, 0xAB, 256);
    EXPECT_EQ(UCC_OK, ucc_mc_free(h));
}

/* ucc_mc_get_mem_attr dispatches to user component mem_query */
UCC_TEST_F(test_mc_user_component, mem_query_dispatch)
{
    ucc_memory_type_t mt;
    ucc_mem_attr_t    mem_attr;
    char              buf[64]; /* mock mem_query ignores the pointer value */

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt));
    g_assigned_type_a = mt;

    mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;
    ASSERT_EQ(UCC_OK, ucc_mc_get_mem_attr(buf, &mem_attr));
    EXPECT_EQ(mt, mem_attr.mem_type);
}

/* After ucc_mc_finalize, user component registry is cleared and type
 * counter resets so the next registration starts at LAST+1 again */
UCC_TEST_F(test_mc_user_component, finalize_resets_registry)
{
    ucc_memory_type_t mt_before, mt_after;

    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt_before));
    EXPECT_EQ(UCC_OK, ucc_mc_available(mt_before));

    /* TearDown calls ucc_mc_finalize which calls finalize_all internally.
     * Call it early here so we can inspect state before TearDown. */
    ucc_mc_finalize();

    EXPECT_NE(UCC_OK, ucc_mc_available(mt_before));

    /* Re-init and re-register: type counter should have reset */
    ucc_mc_params_t mc_params = {.thread_mode = UCC_THREAD_SINGLE};
    ASSERT_EQ(UCC_OK, ucc_mc_init(&mc_params));
    make_mock(&mc_a, "mock_a", mock_mem_query_a);
    ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt_after));
    EXPECT_EQ(mt_before, mt_after);
    /* TearDown will call ucc_mc_finalize again; that's safe */
}

/* -------------------------------------------------------------------------
 * Load-from-path tests
 * These tests exercise ucc_components_load_user_component() directly.
 * ------------------------------------------------------------------------- */

class test_mc_user_component_load : public ucc::test {
protected:
    int orig_n_components;

    virtual void SetUp() override
    {
        ucc::test::SetUp();
        ucc_constructor();
        ucc_mc_params_t mc_params = {.thread_mode = UCC_THREAD_SINGLE};
        ucc_mc_init(&mc_params);
        orig_n_components = ucc_global_config.mc_framework.n_components;
    }

    virtual void TearDown() override
    {
        ucc_mc_finalize();
        /* Trim any user-loaded components so they don't pollute later tests. */
        ucc_global_config.mc_framework.n_components = orig_n_components;
        ucc_global_config.mc_framework.names.count  = orig_n_components;
        ucc::test::TearDown();
    }
};

/* Empty path is a no-op and returns UCC_OK */
UCC_TEST_F(test_mc_user_component_load, empty_path_is_noop)
{
    EXPECT_EQ(UCC_OK,
              ucc_components_load_user_component("", "mc",
                                                 &ucc_global_config.mc_framework));
    EXPECT_EQ(orig_n_components,
              ucc_global_config.mc_framework.n_components);
}

/* A path with no matching .so files returns UCC_ERR_NOT_FOUND */
UCC_TEST_F(test_mc_user_component_load, no_matching_so_returns_not_found)
{
    EXPECT_EQ(UCC_ERR_NOT_FOUND,
              ucc_components_load_user_component("/tmp", "mc",
                                                 &ucc_global_config.mc_framework));
    EXPECT_EQ(orig_n_components,
              ucc_global_config.mc_framework.n_components);
}

#ifdef GTEST_MC_USER_LIB_DIR
/* Loading from the directory containing libucc_mc_mock.so discovers the
 * component, appends it to the framework, and records its name correctly. */
UCC_TEST_F(test_mc_user_component_load, loads_mock_component)
{
    ucc_mc_base_t *mc;

    ASSERT_EQ(UCC_OK,
              ucc_components_load_user_component(GTEST_MC_USER_LIB_DIR, "mc",
                                                 &ucc_global_config.mc_framework));

    ASSERT_EQ(orig_n_components + 1,
              ucc_global_config.mc_framework.n_components);

    mc = ucc_derived_of(
        ucc_global_config.mc_framework.components[orig_n_components],
        ucc_mc_base_t);
    EXPECT_STREQ("mock", mc->super.name);
}

/* After loading, ucc_mc_init() initializes the mock component and it becomes
 * available under its dynamically assigned memory type. */
UCC_TEST_F(test_mc_user_component_load, loaded_component_initializes)
{
    ucc_mc_base_t    *mc;
    ucc_memory_type_t mt;

    ASSERT_EQ(UCC_OK,
              ucc_components_load_user_component(GTEST_MC_USER_LIB_DIR, "mc",
                                                 &ucc_global_config.mc_framework));

    /* Mark it as a user component so ucc_mc_init assigns a dynamic type */
    mc       = ucc_derived_of(
        ucc_global_config.mc_framework.components[orig_n_components],
        ucc_mc_base_t);
    mc->type = UCC_MEMORY_TYPE_LAST;

    ucc_mc_params_t mc_params = {.thread_mode = UCC_THREAD_SINGLE};
    ASSERT_EQ(UCC_OK, ucc_mc_init(&mc_params));

    mt = mc->type;
    EXPECT_GT(mt, UCC_MEMORY_TYPE_LAST);
    EXPECT_EQ(UCC_OK, ucc_mc_available(mt));
    EXPECT_EQ(1, ucc_mc_is_user_component(mt));
    /* TearDown calls ucc_mc_finalize which cleans up */
}
#endif /* GTEST_MC_USER_LIB_DIR */
