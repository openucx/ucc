/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "test_score.h"

UCC_TEST_F(test_score, alloc_free)
{
    ucc_coll_score_t *score;
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    ucc_coll_score_free(score);
}

UCC_TEST_F(test_score, add_range)
{
    ucc_coll_type_t   c = UCC_COLL_TYPE_BARRIER;
    ucc_memory_type_t m = UCC_MEMORY_TYPE_HOST;
    ucc_score_t       value;
    ucc_coll_score_t *score;

    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    /* adding range with 0 score value - should be skipped */
    value = 0;
    EXPECT_EQ(UCC_OK, ucc_coll_score_add_range(score, c, m, 16, 65536, value,
                                               NULL, NULL));
    EXPECT_EQ(0, ucc_list_length(&score->scores[ucc_ilog2(c)][m]));
    value = 10;
    EXPECT_EQ(UCC_OK, ucc_coll_score_add_range(score, c, m, 16, 65536, value,
                                               NULL, NULL));
    EXPECT_EQ(1, ucc_list_length(&score->scores[ucc_ilog2(c)][m]));
    ucc_coll_score_free(score);
}

UCC_TEST_F(test_score, add_range_sorted)
{
    ucc_coll_type_t   c = UCC_COLL_TYPE_BARRIER;
    ucc_memory_type_t m = UCC_MEMORY_TYPE_HOST;
    ucc_score_t       value;
    ucc_coll_score_t *score;

    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
    value = 10;
    EXPECT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, c, m, 50, 60, value, NULL, NULL));
    EXPECT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, c, m, 0, 10, value, NULL, NULL));
    EXPECT_EQ(UCC_OK, ucc_coll_score_add_range(score, c, m, 100, 1000, value,
                                               NULL, NULL));
    EXPECT_EQ(UCC_OK,
              ucc_coll_score_add_range(score, c, m, 20, 40, value, NULL, NULL));
    ucc_list_link_t *list = &score->scores[ucc_ilog2(c)][m];
    EXPECT_EQ(4, ucc_list_length(list));
    ucc_msg_range_t *range;
    size_t           expected[4] = {0, 20, 50, 100};
    size_t *         e           = expected;
    ucc_list_for_each(range, list, list_elem)
    {
        EXPECT_EQ(*e, range->start);
        e++;
    }
    ucc_coll_score_free(score);
}


class test_score_merge : public test_score {
  public:
    ucc_coll_score_t *score1;
    ucc_coll_score_t *score2;
    ucc_coll_score_t *merge;
    test_score_merge()
    {
        EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score1));
        EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score2));
    }
    ~test_score_merge()
    {
        ucc_coll_score_free(merge);
    }
};

ucc_status_t test_score::check_range(ucc_coll_score_t *   score,
                                     ucc_coll_type_t      c,
                                     ucc_memory_type_t    m,
                                     std::vector<range_t> check)
{
    ucc_msg_range_t *range;
    ucc_list_link_t *list = &score->scores[ucc_ilog2(c)][m];
    auto             r    = check.begin();
    if (check.size() != ucc_list_length(list)) {
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_list_for_each(range, list, list_elem)
    {
        if (range->start != std::get<0>(*r) || range->end != std::get<1>(*r) ||
            range->score != std::get<2>(*r)) {
            return UCC_ERR_NO_MESSAGE;
        }
        r++;
    }
    return UCC_OK;
}

void test_score::init_score(ucc_coll_score_t *   score,
                            std::vector<range_t> v, ucc_coll_type_t c)
{
    for (auto &r : v) {
        EXPECT_EQ(UCC_OK, ucc_coll_score_add_range(
                      score, c, UCC_MEMORY_TYPE_HOST, std::get<0>(r),
                      std::get<1>(r), std::get<2>(r), NULL, NULL));
    }
}

UCC_TEST_F(test_score_merge, non_overlap)
{
    init_score(score1, RLIST({RANGE(0, 10, 10), RANGE(40, 50, 5)}),
               UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(10, 20, 100), RANGE(30, 35, 1)}),
               UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 10, 10), RANGE(10, 20, 100),
                                 RANGE(30, 35, 1), RANGE(40, 50, 5)})));
}

UCC_TEST_F(test_score_merge, overlap_single)
{
    init_score(score1, RLIST({RANGE(0, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(50, 150, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 10), RANGE(50, 150, 100)})));
    ucc_coll_score_free(merge);

    init_score(score1, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(50, 150, 10)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 100, 100), RANGE(100, 150, 10)})));
}

UCC_TEST_F(test_score_merge, inclusive)
{
    init_score(score1, RLIST({RANGE(0, 90, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(30, 60, 10)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 90, 100)})));
    ucc_coll_score_free(merge);

    init_score(score1, RLIST({RANGE(0, 90, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(30, 60, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 30, 10), RANGE(30, 60, 100),
                                 RANGE(60, 90, 10)})));
}

UCC_TEST_F(test_score_merge, same_start)
{
    init_score(score1, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(0, 50, 10)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 100, 100)})));
    ucc_coll_score_free(merge);

    init_score(score1, RLIST({RANGE(0, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(0, 50, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 100), RANGE(50, 100, 10)})));
    ucc_coll_score_free(merge);

    init_score(score1, RLIST({RANGE(1, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(1, 50, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(1, 50, 100), RANGE(50, 100, 10)})));
}

UCC_TEST_F(test_score_merge, 1_overlaps_many)
{
    init_score(score1, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2,
               RLIST({RANGE(10, 20, 10), RANGE(30, 40, 10), RANGE(60, 70, 10)}),
               UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 100, 100)})));
    ucc_coll_score_free(merge);

    init_score(score1, RLIST({RANGE(0, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(
        score2,
        RLIST({RANGE(10, 20, 100), RANGE(30, 40, 100), RANGE(60, 70, 5)}),
        UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 10, 10), RANGE(10, 20, 100),
                                 RANGE(20, 30, 10), RANGE(30, 40, 100),
                                 RANGE(40, 100, 10)})));
}

UCC_TEST_F(test_score_merge, same_score)
{
    init_score(score1, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(score2, RLIST({RANGE(100, 200, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_merge(score1, score2, &merge, 1));
    EXPECT_EQ(UCC_OK,
              check_range(merge, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 200, 100)})));
}
