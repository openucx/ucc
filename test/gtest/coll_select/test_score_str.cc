/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
extern "C" {
#include "coll_select/coll_select.h"
}
#include <common/test.h>
#include <string>
#include <vector>

class test_score_str : public ucc::test {
};

#define SCORE(_score, _ct, _mt)                                                \
    ({                                                                         \
        ucc_list_link_t *l =                                                   \
            _score->scores[ucc_ilog2(UCC_COLL_TYPE_##_ct)]                     \
                          [UCC_MEMORY_TYPE_##_mt]                              \
                              .next;                                           \
        ucc_msg_range_t *range =                                               \
            ucc_container_of(l, ucc_msg_range_t, list_elem);                   \
        range->score;                                                          \
    })

UCC_TEST_F(test_score_str, check_valid)
{
    std::string       str = "alltoall:cuda:10";
    ucc_coll_score_t *score;
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0));
    EXPECT_EQ(10, SCORE(score, ALLTOALL, CUDA));
    ucc_coll_score_free(score);

    str = "host,Cuda:Bcast,SCATTER:10";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0));
    EXPECT_EQ(10, SCORE(score, BCAST, CUDA));
    EXPECT_EQ(10, SCORE(score, BCAST, HOST));
    EXPECT_EQ(10, SCORE(score, SCATTER, CUDA));
    EXPECT_EQ(10, SCORE(score, SCATTER, HOST));
    ucc_coll_score_free(score);

    str = "inf:gatHerv";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0));
    EXPECT_EQ(UCC_SCORE_MAX, SCORE(score, GATHERV, ROCM));
    EXPECT_EQ(UCC_SCORE_MAX, SCORE(score, GATHERV, HOST));
    ucc_coll_score_free(score);

    str = "alltoall,bCAst:hOst:10;scatter:inf;reduce:1";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0));
    EXPECT_EQ(10, SCORE(score, BCAST, HOST));
    EXPECT_EQ(UCC_SCORE_MAX, SCORE(score, SCATTER, HOST));
    EXPECT_EQ(UCC_SCORE_MAX, SCORE(score, SCATTER, CUDA_MANAGED));
    EXPECT_EQ(1, SCORE(score, REDUCE, HOST));
    ucc_coll_score_free(score);
}

UCC_TEST_F(test_score_str, check_invalid)
{
    std::string       str = "alltoallll:cuda:10";
    ucc_coll_score_t *score;
    testing::internal::CaptureStdout();
    EXPECT_NE(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0));
    testing::internal::GetCapturedStdout();
}
