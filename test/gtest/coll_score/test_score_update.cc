/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * RESERVED.  See file LICENSE for terms.
 */
#include "test_score.h"

class test_score_update : public test_score {
  public:
    ucc_coll_score_t *score;
    ucc_coll_score_t *update;

    test_score_update()
    {
        EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
        EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&update));
    }
    ~test_score_update()
    {
        ucc_coll_score_free(score);
        ucc_coll_score_free(update);
    }
    void reset() {
        ucc_coll_score_free(score);
        ucc_coll_score_free(update);
        EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&score));
        EXPECT_EQ(UCC_OK, ucc_coll_score_alloc(&update));
    }
};

UCC_TEST_F(test_score_update, non_overlap)
{
    init_score(score, RLIST({RANGE(0, 10, 10), RANGE(40, 50, 5)}),
               UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(10, 20, 100), RANGE(30, 35, 1)}),
               UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 10, 10), RANGE(40, 50, 5)})));
}

UCC_TEST_F(test_score_update, overlap_single)
{
    init_score(score, RLIST({RANGE(0, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(50, 150, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 10), RANGE(50, 100, 100)})));
    reset();

    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(50, 150, 10)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 100), RANGE(50, 100, 10)})));
}

UCC_TEST_F(test_score_update, inclusive)
{
    init_score(score, RLIST({RANGE(0, 90, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(30, 60, 10)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 30, 100), RANGE(30, 60, 10), RANGE(60, 90, 100)})));
    reset();

    init_score(score, RLIST({RANGE(0, 90, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(30, 60, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 30, 10), RANGE(30, 60, 100),
                                 RANGE(60, 90, 10)})));
}

UCC_TEST_F(test_score_update, same_start)
{
    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(0, 50, 10)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 10), RANGE(50, 100, 100)})));
    reset();

    init_score(score, RLIST({RANGE(0, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(0, 50, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 100), RANGE(50, 100, 10)})));
    reset();

    init_score(score, RLIST({RANGE(1, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(1, 50, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(1, 50, 100), RANGE(50, 100, 10)})));
}

UCC_TEST_F(test_score_update, 1_overlaps_many)
{
    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(update,
               RLIST({RANGE(10, 20, 10), RANGE(30, 40, 20), RANGE(60, 70, 30)}),
               UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 10, 100), RANGE(10, 20, 10),
                                 RANGE(20, 30, 100), RANGE(30, 40, 20),
                                 RANGE(40, 60, 100), RANGE(60, 70, 30), RANGE(70, 100, 100)})));
    reset();

    init_score(score, RLIST({RANGE(0, 100, 10)}), UCC_COLL_TYPE_BARRIER);
    init_score(
        update,
        RLIST({RANGE(10, 20, 100), RANGE(30, 40, 100), RANGE(60, 70, 5)}),
        UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 10, 10), RANGE(10, 20, 100),
                                 RANGE(20, 30, 10), RANGE(30, 40, 100),
                                 RANGE(40, 60, 10), RANGE(60, 70, 5), RANGE(70, 100, 10)})));
}

UCC_TEST_F(test_score_update, same_score)
{
    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER);
    init_score(update, RLIST({RANGE(100, 200, 100)}), UCC_COLL_TYPE_BARRIER);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 100, 100)})));
}

UCC_TEST_F(test_score_update, non_overlap_2)
{
    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER, 0x1,
               0x1);
    init_score(update, RLIST({RANGE(300, 400, 100)}), UCC_COLL_TYPE_BARRIER,
               0x2, 0x2);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 100, 100), RANGE(300, 400, 100)})));
}

UCC_TEST_F(test_score_update, init_reset)
{
    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER, 0x1,
               0x1);
    init_score(update, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER, 0x2,
               0x2);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 100, 100)})));
    EXPECT_EQ(0x2, (uint64_t)(FIRST_RANGE(score, BARRIER, HOST)->super.init));

    reset();

    init_score(score, RLIST({RANGE(0, 100, 100)}), UCC_COLL_TYPE_BARRIER, 0x1,
               0x1);
    init_score(update, RLIST({RANGE(50, 150, 50)}), UCC_COLL_TYPE_BARRIER, 0x2,
               0x2);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 100), RANGE(50, 100, 50),
                                 RANGE(100, 150, 50)})));
    EXPECT_EQ(0x1, (uint64_t)(FIRST_RANGE(score, BARRIER, HOST)->super.init));

    reset();
    init_score(score, RLIST({RANGE(50, 150, 100)}), UCC_COLL_TYPE_BARRIER, 0x1,
               0x1);
    init_score(update, RLIST({RANGE(0, 100, 50)}), UCC_COLL_TYPE_BARRIER, 0x2,
               0x2);
    EXPECT_EQ(UCC_OK, ucc_coll_score_update(score, update, 0));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(0, 50, 50), RANGE(50, 100, 50),
                                 RANGE(100, 150, 100)})));
    EXPECT_EQ(0x2, (uint64_t)(FIRST_RANGE(score, BARRIER, HOST)->super.init));
}
