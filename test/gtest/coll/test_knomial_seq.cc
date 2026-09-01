/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include "utils/ucc_coll_utils.h"
#include "coll_patterns/sra_knomial.h"
}

#include <common/test.h>
#include <vector>

class test_knomial_seq : public ucc::test {
  protected:
    static void expect_exact_pattern(
        ucc_rank_t size, const std::vector<ucc_kn_radix_t> &radices)
    {
        ucc_kn_radix_seq_t radix_seq = {};

        ASSERT_LE(radices.size(), UCC_KN_MAX_RADIX_PHASES);
        radix_seq.radices   = radices.data();
        radix_seq.n_radices = static_cast<uint8_t>(radices.size());
        for (ucc_rank_t rank = 0; rank < size; rank++) {
            ucc_knomial_pattern_t p;
            ucc_rank_t            phase_size = 1;

            ucc_kn_ag_pattern_init(size, rank, &radix_seq, size, &p);
            ASSERT_EQ(radix_seq.radices, p.radix_seq.radices);
            ASSERT_EQ(radix_seq.n_radices, p.radix_seq.n_radices);
            ASSERT_EQ(radices.size(), p.n_iters);
            ASSERT_EQ(size, p.full_pow_size);
            ASSERT_EQ(0, p.n_extra);
            ASSERT_EQ(KN_NODE_BASE, p.node_type);
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

            ucc_kn_ag_pattern_init(size, rank, &radix_seq, size, &p);
            EXPECT_EQ(0, p.iteration);
            EXPECT_EQ(ucc_kn_radix_seq_get(&radix_seq, 0), p.radix);
            EXPECT_EQ(radix_seq.n_radices, p.n_iters);
        }
    }
};

UCC_TEST_F(test_knomial_seq, fixed_pattern_preserves_legacy_layout)
{
    const ucc_rank_t     sizes[]   = {16, 48, 72, 96};
    const ucc_kn_radix_t radices[] = {2, 3, 4, 6, 8};

    for (auto size : sizes) {
        for (auto radix : radices) {
            for (ucc_rank_t rank = 0; rank < size; rank++) {
                ucc_knomial_pattern_t p;
                ucc_kn_radix_seq_t    radix_seq =
                    ucc_kn_radix_seq_from_radix(radix);
                ucc_rank_t            legacy_radix_pow = 1;

                ucc_kn_ag_pattern_init(size, rank, &radix_seq, size, &p);
                ASSERT_EQ(1, p.radix_seq.n_radices);
                ASSERT_EQ(radix,
                          ucc_kn_radix_seq_get(&p.radix_seq, 0));
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

UCC_TEST_F(test_knomial_seq, exact_peer_and_segment_layout)
{
    expect_exact_pattern(48, {8, 6});
    expect_exact_pattern(16, {8, 2});
    expect_exact_pattern(16, {2, 8});
    expect_exact_pattern(72, {8, 9});
    expect_exact_pattern(72, {3, 3, 2, 2, 2});
    expect_exact_pattern(96, {4, 4, 6});
}

UCC_TEST_F(test_knomial_seq, validation)
{
    const ucc_kn_radix_t valid_radices[]    = {8, 6};
    const ucc_kn_radix_t zero_radices[]     = {8, 0, 6};
    const ucc_kn_radix_t overflow_radices[] = {UINT16_MAX, UINT16_MAX, 2};
    ucc_kn_radix_seq_t   seq                = {};

    EXPECT_TRUE(ucc_kn_radix_seq_is_valid(&seq, 48));

    seq = ucc_kn_radix_seq_from_radix(0);
    EXPECT_FALSE(ucc_kn_radix_seq_is_valid(&seq, 48));

    seq.radices   = valid_radices;
    seq.n_radices = 2;
    EXPECT_TRUE(ucc_kn_radix_seq_is_valid(&seq, 48));
    EXPECT_FALSE(ucc_kn_radix_seq_is_valid(&seq, 64));

    seq.radices   = zero_radices;
    seq.n_radices = 3;
    EXPECT_FALSE(ucc_kn_radix_seq_is_valid(&seq, 48));

    seq.radices   = overflow_radices;
    seq.n_radices = 3;
    EXPECT_FALSE(ucc_kn_radix_seq_is_valid(&seq, UINT32_MAX));

    seq.radices   = nullptr;
    seq.n_radices = 2;
    EXPECT_FALSE(ucc_kn_radix_seq_is_valid(&seq, 48));

    seq.n_radices = UCC_KN_MAX_RADIX_PHASES + 1;
    EXPECT_FALSE(ucc_kn_radix_seq_is_valid(&seq, 48));
}
