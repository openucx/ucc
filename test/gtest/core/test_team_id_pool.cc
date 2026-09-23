/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "common/test_ucc.h"
extern "C" {
#include "core/ucc_team.h"
}

class test_team_id_pool : public ucc::test {
};

/* Team-id pool bit helpers must be exact inverses at word boundaries. */
UCC_TEST_F(test_team_id_pool, bit_boundaries)
{
    const int ids[] = {1, 63, 64, 65, 127, 128, 129, 191, 192};

    for (size_t k = 0; k < sizeof(ids) / sizeof(ids[0]); k++) {
        int      id      = ids[k];
        uint64_t pool[4] = {0, 0, 0, 0}; /* covers ids 1..256 */

        ucc_team_id_pool_set_bit(pool, id);

        int set_words = 0;
        for (int w = 0; w < 4; w++) {
            if (pool[w]) {
                set_words++;
            }
        }
        EXPECT_EQ(1, set_words) << "id " << id << " set bits in multiple words";

        int found = 0, pos = 0;
        for (int w = 0; w < 4; w++) {
            if ((pos = ucc_team_id_pool_ffs_clear(&pool[w])) > 0) {
                found = w * 64 + pos;
                break;
            }
        }
        EXPECT_EQ(id, found)
            << "released id " << id << " re-scanned as " << found;

        for (int w = 0; w < 4; w++) {
            EXPECT_EQ((uint64_t)0, pool[w])
                << "id " << id << " left residue in word " << w;
        }
    }
}
