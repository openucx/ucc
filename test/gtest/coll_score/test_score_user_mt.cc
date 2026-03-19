/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

/**
 * Tests for user memory type integration with the scoring system.
 *
 * These tests exercise the Phase 1 infrastructure change: ucc_coll_score_t
 * uses a dynamically-sized flat array so that user memory types
 * (>= UCC_MEMORY_TYPE_LAST) are first-class entries in score structures,
 * the merge path, and map lookup.
 *
 * Each test calls ucc_constructor() + ucc_mc_init() in SetUp and registers
 * in-process mock MC components so that ucc_mc_total_mem_types() returns a
 * value that covers user type indices.
 */

extern "C" {
#include "coll_score/ucc_coll_score.h"
#include "components/mc/ucc_mc.h"
#include "components/mc/ucc_mc_user_component.h"
}
#include "coll_score/test_score.h"
#include <cstring>
#include <cstdlib>

/* -------------------------------------------------------------------------
 * Minimal mock MC callbacks (scoped to this translation unit)
 * ------------------------------------------------------------------------- */

static ucc_status_t usrmt_mock_init(const ucc_mc_params_t *)
{
    return UCC_OK;
}

static ucc_status_t usrmt_mock_get_attr(ucc_mc_attr_t *attr)
{
    attr->thread_mode = UCC_THREAD_SINGLE;
    return UCC_OK;
}

static ucc_status_t usrmt_mock_finalize(void)
{
    return UCC_OK;
}

static ucc_status_t usrmt_mock_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                         size_t                   size,
                                         ucc_memory_type_t        mt)
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

static ucc_status_t usrmt_mock_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    free(h_ptr);
    return UCC_OK;
}

static ucc_status_t usrmt_mock_memcpy(void *dst, const void *src, size_t len,
                                      ucc_memory_type_t, ucc_memory_type_t)
{
    memcpy(dst, src, len);
    return UCC_OK;
}

static ucc_status_t usrmt_mock_memset(void *ptr, int val, size_t len)
{
    memset(ptr, val, len);
    return UCC_OK;
}

/* -------------------------------------------------------------------------
 * Fixture
 * ------------------------------------------------------------------------- */

class test_score_user_mt : public ucc::test {
protected:
    ucc_mc_base_t     mc_a, mc_b;
    ucc_memory_type_t mt_a, mt_b;

    void make_mock(ucc_mc_base_t *mc, const char *name)
    {
        memset(mc, 0, sizeof(*mc));
        mc->super.name    = name;
        mc->ee_type       = UCC_EE_CPU_THREAD;
        mc->type          = UCC_MEMORY_TYPE_LAST;
        mc->ref_cnt       = 0;
        mc->init          = usrmt_mock_init;
        mc->get_attr      = usrmt_mock_get_attr;
        mc->finalize      = usrmt_mock_finalize;
        mc->ops.mem_alloc = usrmt_mock_mem_alloc;
        mc->ops.mem_free  = usrmt_mock_mem_free;
        mc->ops.memcpy    = usrmt_mock_memcpy;
        mc->ops.memset    = usrmt_mock_memset;
    }

    virtual void SetUp() override
    {
        ucc::test::SetUp();
        ucc_constructor();
        ucc_mc_params_t mc_params = {.thread_mode = UCC_THREAD_SINGLE};
        ucc_mc_init(&mc_params);
        make_mock(&mc_a, "usrmt_a");
        make_mock(&mc_b, "usrmt_b");
        ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_a, &mt_a));
        ASSERT_EQ(UCC_OK, ucc_mc_user_component_register(&mc_b, &mt_b));
        mc_a.type = mt_a;
        mc_b.type = mt_b;
    }

    virtual void TearDown() override
    {
        ucc_mc_finalize();
        ucc::test::TearDown();
    }
};

/* -------------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------------- */

/* Score allocated after user MC registration has n_mem_types large enough
 * to index both user types without going out of bounds. */
UCC_TEST_F(test_score_user_mt, alloc_covers_user_types)
{
    ucc_coll_score_t *score;
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    EXPECT_GE(score->n_mem_types, (int)mt_a + 1);
    EXPECT_GE(score->n_mem_types, (int)mt_b + 1);
    ucc_coll_score_free(score);
}

/* ucc_coll_score_add_range succeeds for a user memory type and the range
 * is retrievable via ucc_score_list. */
UCC_TEST_F(test_score_user_mt, add_range_user_mt)
{
    ucc_coll_score_t *score;
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    EXPECT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLREDUCE, mt_a,
                                       0, UCC_MSG_MAX, 10, NULL, NULL));
    EXPECT_EQ(1, ucc_list_length(
                     ucc_score_list(score, ucc_ilog2(UCC_COLL_TYPE_ALLREDUCE),
                                    mt_a)));
    ucc_coll_score_free(score);
}

/* Built-in memory type ranges are unaffected when user types are also added. */
UCC_TEST_F(test_score_user_mt, builtin_mt_unaffected)
{
    ucc_coll_score_t *score;
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, UCC_COLL_TYPE_BARRIER,
                                       UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX,
                                       50, NULL, NULL));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, UCC_COLL_TYPE_BARRIER, mt_a,
                                       0, UCC_MSG_MAX, 20, NULL, NULL));
    EXPECT_EQ(1,
              ucc_list_length(ucc_score_list(
                  score, ucc_ilog2(UCC_COLL_TYPE_BARRIER),
                  UCC_MEMORY_TYPE_HOST)));
    EXPECT_EQ(1,
              ucc_list_length(ucc_score_list(
                  score, ucc_ilog2(UCC_COLL_TYPE_BARRIER), mt_a)));
    ucc_coll_score_free(score);
}

/* Two user memory types are stored independently; neither pollutes the other
 * or the HOST slot. */
UCC_TEST_F(test_score_user_mt, two_user_mts_independent)
{
    ucc_coll_score_t *score;
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, UCC_COLL_TYPE_BCAST, mt_a,
                                       0, UCC_MSG_MAX, 10, NULL, NULL));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, UCC_COLL_TYPE_BCAST, mt_b,
                                       0, UCC_MSG_MAX, 20, NULL, NULL));
    EXPECT_EQ(1, ucc_list_length(
                     ucc_score_list(score, ucc_ilog2(UCC_COLL_TYPE_BCAST),
                                    mt_a)));
    EXPECT_EQ(1, ucc_list_length(
                     ucc_score_list(score, ucc_ilog2(UCC_COLL_TYPE_BCAST),
                                    mt_b)));
    EXPECT_EQ(0, ucc_list_length(
                     ucc_score_list(score, ucc_ilog2(UCC_COLL_TYPE_BCAST),
                                    UCC_MEMORY_TYPE_HOST)));
    ucc_coll_score_free(score);
}

/* Merging two scores that each cover a non-overlapping sub-range of a user
 * memory type produces a merged score with both ranges present. */
UCC_TEST_F(test_score_user_mt, merge_user_mt)
{
    ucc_coll_score_t *s1, *s2, *merged;
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&s1));
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&s2));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(s1, UCC_COLL_TYPE_ALLREDUCE, mt_a,
                                       0, 1024, 50, NULL, NULL));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(s2, UCC_COLL_TYPE_ALLREDUCE, mt_a,
                                       1024, UCC_MSG_MAX, 30, NULL, NULL));
    ASSERT_EQ(UCC_OK, ucc_coll_score_merge(s1, s2, &merged, 1));
    EXPECT_EQ(2, ucc_list_length(
                     ucc_score_list(merged,
                                    ucc_ilog2(UCC_COLL_TYPE_ALLREDUCE),
                                    mt_a)));
    ucc_coll_score_free(merged);
}

/* Merging with a user type range on one side and nothing on the other
 * produces a merged score that contains the range. */
UCC_TEST_F(test_score_user_mt, merge_user_mt_one_sided)
{
    ucc_coll_score_t *s1, *s2, *merged;
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&s1));
    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&s2));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(s1, UCC_COLL_TYPE_ALLREDUCE, mt_a,
                                       0, UCC_MSG_MAX, 10, NULL, NULL));
    /* s2 has no range for mt_a */
    ASSERT_EQ(UCC_OK, ucc_coll_score_merge(s1, s2, &merged, 1));
    EXPECT_EQ(1, ucc_list_length(
                     ucc_score_list(merged,
                                    ucc_ilog2(UCC_COLL_TYPE_ALLREDUCE),
                                    mt_a)));
    ucc_coll_score_free(merged);
}

/* ucc_coll_score_build_map succeeds when the score contains user MT ranges,
 * and the map can be freed without error. */
UCC_TEST_F(test_score_user_mt, build_map_user_mt)
{
    ucc_coll_score_t *score;
    ucc_score_map_t  *map;

    ASSERT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    ASSERT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, UCC_COLL_TYPE_ALLREDUCE, mt_a,
                                       0, UCC_MSG_MAX, 10, NULL, NULL));
    ASSERT_EQ(UCC_OK, ucc_coll_score_build_map(score, &map));
    /* score ownership transferred to map; just free the map */
    ucc_free(map);
}
