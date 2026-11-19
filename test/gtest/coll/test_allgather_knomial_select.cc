/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include "components/tl/ucp/allgather/allgather_knomial_select.h"
}

#include <common/test.h>
#include <vector>

class test_allgather_knomial_select : public ucc::test {
  protected:
    static void expect_selection(
        ucc_rank_t size, size_t msg_size,
        const std::vector<ucc_kn_radix_t> &expected)
    {
        ucc_kn_radix_t     radices[UCC_KN_MAX_RADIX_PHASES];
        ucc_kn_radix_seq_t radix_seq;

        ASSERT_TRUE(ucc_tl_ucp_allgather_knomial_select_radix_seq(
            size, msg_size, radices, &radix_seq));
        ASSERT_EQ(expected.size(), radix_seq.n_radices);
        if (expected.size() == 1) {
            EXPECT_EQ(nullptr, radix_seq.radices);
        } else {
            EXPECT_EQ(radices, radix_seq.radices);
        }
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(expected[i], ucc_kn_radix_seq_get(&radix_seq, i));
        }
    }
};

UCC_TEST_F(test_allgather_knomial_select, minimizes_small_message_fanout)
{
    expect_selection(48, 1 << 20, {8, 6});
    expect_selection(60, 1 << 20, {4, 5, 3});
    expect_selection(64, 1 << 20, {8});
    expect_selection(72, 1 << 20, {8, 9});
    expect_selection(96, 1 << 20, {4, 4, 6});
    expect_selection(
        128, UCC_TL_UCP_ALLGATHER_KN_LARGE_MSG_SIZE - 1, {8, 4, 4});
}

UCC_TEST_F(test_allgather_knomial_select, uses_r2_for_large_messages)
{
    expect_selection(
        96, UCC_TL_UCP_ALLGATHER_KN_LARGE_MSG_SIZE, {2, 2, 2, 2, 2, 3});
    expect_selection(128, UCC_TL_UCP_ALLGATHER_KN_LARGE_MSG_SIZE, {2});
}

UCC_TEST_F(
    test_allgather_knomial_select, handles_odd_factors_and_unsupported_sizes)
{
    const ucc_rank_t   unsupported_sizes[] = {0, 1, 11};
    ucc_kn_radix_t     radices[UCC_KN_MAX_RADIX_PHASES];
    ucc_kn_radix_seq_t radix_seq;

    expect_selection(25, 1 << 20, {5});
    expect_selection(35, 1 << 20, {7, 5});
    for (auto size : unsupported_sizes) {
        EXPECT_FALSE(ucc_tl_ucp_allgather_knomial_select_radix_seq(
            size, 1 << 20, radices, &radix_seq));
        EXPECT_EQ(0, radix_seq.n_radices);
        EXPECT_EQ(nullptr, radix_seq.radices);
    }
}
