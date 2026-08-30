/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "common/gtest.h"
#include "components/tl/ucp/alltoall/alltoall_pairwise_num_posts.h"

TEST(alltoall_pairwise_num_posts, automatic_boundaries)
{
    struct test_case {
        ucc_rank_t tsize;
        size_t     peer_size;
        ucc_rank_t expected;
    };
    const test_case cases[] = {
        {64, 0, 32},
        {64, 64 * 1024, 32},
        {64, 64 * 1024 + 1, 16},
        {64, 1024 * 1024, 16},
        {16, 1024 * 1024 + 1, 16},
        {17, 1024 * 1024 + 1, 8},
        {64, 4 * 1024 * 1024, 8},
        {64, 4 * 1024 * 1024 + 1, 4},
        {64, 8 * 1024 * 1024, 4},
        {8, 4 * 1024 * 1024 + 1, 8},
        {9, 4 * 1024 * 1024 + 1, 4},
        {8, 8 * 1024 * 1024 + 1, 8},
        {9, 8 * 1024 * 1024, 4},
        {9, 8 * 1024 * 1024 + 1, 1},
        {16, 8 * 1024 * 1024 + 1, 1},
        {32, 8 * 1024 * 1024 + 1, 1},
        {33, 8 * 1024 * 1024 + 1, 1},
    };

    for (const auto &c : cases) {
        EXPECT_EQ(c.expected, ucc_tl_ucp_alltoall_pairwise_auto_num_posts(
                                  c.tsize, c.peer_size));
    }
}

TEST(alltoall_pairwise_num_posts, valid_and_nonincreasing)
{
    const size_t peer_sizes[] = {
        0,
        1,
        64 * 1024,
        64 * 1024 + 1,
        1024 * 1024,
        1024 * 1024 + 1,
        4 * 1024 * 1024,
        4 * 1024 * 1024 + 1,
        8 * 1024 * 1024,
        8 * 1024 * 1024 + 1,
    };

    for (ucc_rank_t tsize = 1; tsize <= 128; tsize++) {
        ucc_rank_t previous = tsize;

        for (size_t peer_size : peer_sizes) {
            ucc_rank_t posts = ucc_tl_ucp_alltoall_pairwise_auto_num_posts(
                tsize, peer_size);

            EXPECT_GE(posts, 1);
            EXPECT_LE(posts, tsize);
            EXPECT_LE(posts, previous)
                << "team size " << tsize << ", peer size " << peer_size;
            previous = posts;
        }
    }
}
