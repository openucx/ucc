/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include "utils/ucc_coll_utils.h"
#include "coll_patterns/sra_knomial.h"

ucc_status_t ucc_tl_ucp_allgather_knomial_parse_radices(
    const char *value, ucc_rank_t team_size, ucc_kn_radix_t *radices,
    uint8_t *nradices);

int ucc_tl_ucp_allgather_knomial_select_radices(
    ucc_rank_t team_size, size_t msg_size, ucc_kn_radix_t *radix,
    ucc_kn_radix_t *radices, uint8_t *nradices);
}

#include <common/test.h>
#include <sstream>
#include <vector>

class test_knomial_schedule : public ucc::test {
  protected:
    static void expect_parse_status(
        const char *value, ucc_rank_t size, ucc_status_t expected)
    {
        ucc_kn_radix_t radices[UCC_KN_MAX_RADIX_PHASES];
        uint8_t        nradices;

        EXPECT_EQ(
            expected,
            ucc_tl_ucp_allgather_knomial_parse_radices(
                value, size, radices, &nradices));
    }

    static void expect_valid(
        const char *value, ucc_rank_t size,
        const std::vector<ucc_kn_radix_t> &expected)
    {
        ucc_kn_radix_t radices[UCC_KN_MAX_RADIX_PHASES];
        uint8_t        nradices;

        ASSERT_EQ(
            UCC_OK,
            ucc_tl_ucp_allgather_knomial_parse_radices(
                value, size, radices, &nradices));
        ASSERT_EQ(expected.size(), nradices);
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(expected[i], radices[i]);
        }
    }

    static void expect_mixed_pattern(
        ucc_rank_t size, const std::vector<ucc_kn_radix_t> &radices)
    {
        for (ucc_rank_t rank = 0; rank < size; rank++) {
            ucc_knomial_pattern_t p;
            ucc_rank_t            phase_size = 1;

            ucc_kn_ag_pattern_init_mixed(
                size, rank, size, radices.data(), radices.size(), &p);
            ASSERT_TRUE(p.is_mixed);
            ASSERT_EQ(radices.size(), p.n_iters);
            ASSERT_EQ(0, p.n_extra);
            for (size_t phase = 0; phase < radices.size(); phase++) {
                ucc_kn_radix_t radix      = radices[phase];
                ucc_rank_t     group_size = phase_size * radix;
                size_t         count;
                ptrdiff_t      offset;

                EXPECT_EQ(radix, p.radix);
                EXPECT_EQ(radix, ucc_kn_compute_step_radix(&p));
                ucc_kn_ag_pattern_peer_seg(rank, &p, &count, &offset);
                EXPECT_EQ(phase_size, count);
                EXPECT_EQ((rank / phase_size) * phase_size, offset);

                for (ucc_kn_radix_t step = 1; step < radix; step++) {
                    ucc_rank_t peer = ucc_knomial_pattern_get_loop_peer(
                        &p, rank, step);
                    ASSERT_NE(UCC_KN_PEER_NULL, peer);
                    EXPECT_EQ(rank / group_size, peer / group_size);
                    EXPECT_EQ(rank % phase_size, peer % phase_size);
                    ucc_kn_ag_pattern_peer_seg(peer, &p, &count, &offset);
                    EXPECT_EQ(phase_size, count);
                    EXPECT_EQ((peer / phase_size) * phase_size, offset);
                }
                ucc_kn_ag_pattern_next_iter(&p);
                phase_size = group_size;
            }
            EXPECT_TRUE(ucc_knomial_pattern_loop_done(&p));
            EXPECT_EQ(size, phase_size);
        }
    }

    static void expect_selection(
        ucc_rank_t size, size_t msg_size, ucc_kn_radix_t expected_fixed_radix,
        const std::vector<ucc_kn_radix_t> &expected_radices)
    {
        ucc_kn_radix_t radices[UCC_KN_MAX_RADIX_PHASES];
        ucc_kn_radix_t radix;
        uint8_t        nradices;

        ASSERT_TRUE(ucc_tl_ucp_allgather_knomial_select_radices(
            size, msg_size, &radix, radices, &nradices));
        ASSERT_EQ(expected_radices.size(), nradices);
        if (nradices == 0) {
            EXPECT_EQ(expected_fixed_radix, radix);
        }
        for (size_t i = 0; i < expected_radices.size(); i++) {
            EXPECT_EQ(expected_radices[i], radices[i]);
        }
    }
};

UCC_TEST_F(test_knomial_schedule, parse_valid_exact_schedules)
{
    expect_valid("8,6", 48, {8, 6});
    expect_valid("8,9", 72, {8, 9});
    expect_valid("3,3,2,2,2", 72, {3, 3, 2, 2, 2});
    expect_valid("4,4,6", 96, {4, 4, 6});
}

UCC_TEST_F(test_knomial_schedule, parse_invalid_tokens_and_radices)
{
    expect_parse_status("", 96, UCC_ERR_NOT_FOUND);
    expect_parse_status("4,x,6", 96, UCC_ERR_INVALID_PARAM);
    expect_parse_status("4,0,6", 96, UCC_ERR_INVALID_PARAM);
    expect_parse_status("4,1,6", 96, UCC_ERR_INVALID_PARAM);
    expect_parse_status("4,4,4", 96, UCC_ERR_INVALID_PARAM);
    expect_parse_status("4,4,6,", 96, UCC_ERR_INVALID_PARAM);
    expect_parse_status("65536", 65536, UCC_ERR_INVALID_PARAM);
    expect_parse_status("65535,65535,2", 2, UCC_ERR_INVALID_PARAM);
}

UCC_TEST_F(test_knomial_schedule, parse_overlong_schedule)
{
    std::ostringstream value;

    for (unsigned i = 0; i <= UCC_KN_MAX_RADIX_PHASES; i++) {
        value << (i ? ",2" : "2");
    }
    expect_parse_status(
        value.str().c_str(), UCC_RANK_MAX, UCC_ERR_INVALID_PARAM);
}

UCC_TEST_F(test_knomial_schedule, fixed_pattern_preserves_legacy_layout)
{
    const ucc_rank_t     sizes[]   = {16, 48, 72, 96};
    const ucc_kn_radix_t radices[] = {2, 3, 4, 6, 8};

    for (auto size : sizes) {
        for (auto radix : radices) {
            for (ucc_rank_t rank = 0; rank < size; rank++) {
                ucc_knomial_pattern_t p;
                ucc_rank_t            legacy_radix_pow = 1;

                ucc_kn_ag_pattern_init(size, rank, radix, size, &p);
                ASSERT_FALSE(p.is_mixed);
                for (uint8_t phase = 0; phase < p.n_iters; phase++) {
                    ucc_rank_t n_full               = size / p.full_pow_size;
                    ucc_rank_t legacy_segment_radix = radix;

                    if (legacy_radix_pow * radix >= size && n_full > 1) {
                        legacy_segment_radix = n_full;
                    }
                    EXPECT_EQ(radix, p.radix);
                    EXPECT_EQ(
                        legacy_segment_radix, ucc_kn_compute_step_radix(&p));
                    if (p.node_type != KN_NODE_EXTRA) {
                        for (ucc_kn_radix_t step = 1; step < radix; step++) {
                            ucc_rank_t
                                loop_rank = ucc_knomial_pattern_loop_rank(
                                    &p, rank);
                            ucc_rank_t step_size = legacy_radix_pow * radix;
                            ucc_rank_t peer      = (loop_rank +
                                               step * legacy_radix_pow) %
                                                  step_size +
                                              ucc_align_down(
                                                  loop_rank, step_size);
                            ucc_rank_t expected =
                                peer >= size - p.n_extra
                                    ? UCC_KN_PEER_NULL
                                    : ucc_knomial_pattern_loop_rank_inv(
                                          &p, peer);
                            EXPECT_EQ(
                                expected,
                                ucc_knomial_pattern_get_loop_peer(
                                    &p, rank, step));
                        }
                    }
                    ucc_kn_ag_pattern_next_iter(&p);
                    legacy_radix_pow *= radix;
                }
            }
        }
    }
}

UCC_TEST_F(test_knomial_schedule, mixed_peer_and_segment_layout)
{
    expect_mixed_pattern(48, {8, 6});
    expect_mixed_pattern(72, {8, 9});
    expect_mixed_pattern(72, {3, 3, 2, 2, 2});
    expect_mixed_pattern(96, {4, 4, 6});
}

UCC_TEST_F(test_knomial_schedule, selector_minimizes_small_message_fanout)
{
    expect_selection(48, 1 << 20, 0, {8, 6});
    expect_selection(60, 1 << 20, 0, {4, 5, 3});
    expect_selection(64, 1 << 20, 8, {});
    expect_selection(72, 1 << 20, 0, {8, 9});
    expect_selection(96, 1 << 20, 0, {4, 4, 6});
    expect_selection(128, (1ull << 30) - 1, 0, {8, 4, 4});
}

UCC_TEST_F(test_knomial_schedule, selector_uses_r2_for_large_messages)
{
    expect_selection(96, 1ull << 30, 0, {2, 2, 2, 2, 2, 3});
    expect_selection(128, 1ull << 30, 2, {});
}

UCC_TEST_F(test_knomial_schedule, selector_handles_odd_factors_and_fallback)
{
    ucc_kn_radix_t radices[UCC_KN_MAX_RADIX_PHASES];
    ucc_kn_radix_t radix;
    uint8_t        nradices;

    expect_selection(25, 1 << 20, 5, {});
    expect_selection(35, 1 << 20, 0, {7, 5});
    EXPECT_FALSE(ucc_tl_ucp_allgather_knomial_select_radices(
        11, 1 << 20, &radix, radices, &nradices));
}
