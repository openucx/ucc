/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "test_score.h"
class test_score_str : public test_score {
};

#define SCORE(_score, _ct, _mt)                                                \
    ({                                                                         \
        ucc_list_link_t *l =                                                   \
            _score->scores[ucc_ilog2(UCC_COLL_TYPE_##_ct)]                     \
                          [UCC_MEMORY_TYPE_##_mt]                              \
                              .next;                                           \
        ucc_msg_range_t *range =                                               \
            ucc_container_of(l, ucc_msg_range_t, super.list_elem);             \
        range->super.score;                                                    \
    })

UCC_TEST_F(test_score_str, check_valid)
{
    std::string       str = "alltoall:cuda:10";
    ucc_coll_score_t *score;

    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(10, SCORE(score, ALLTOALL, CUDA));
    ucc_coll_score_free(score);

    str = "host,Cuda:Bcast,SCATTER:10";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(10, SCORE(score, BCAST, CUDA));
    EXPECT_EQ(10, SCORE(score, BCAST, HOST));
    EXPECT_EQ(10, SCORE(score, SCATTER, CUDA));
    EXPECT_EQ(10, SCORE(score, SCATTER, HOST));
    ucc_coll_score_free(score);

    str = "inf:gatHerv";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(UCC_SCORE_MAX, SCORE(score, GATHERV, ROCM));
    EXPECT_EQ(UCC_SCORE_MAX, SCORE(score, GATHERV, HOST));
    ucc_coll_score_free(score);

    str = "alltoall,bCAst:hOst:10#scatter:inf#reduce:1";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
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
    EXPECT_NE(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    testing::internal::GetCapturedStdout();
}

UCC_TEST_F(test_score_str, check_range_1)
{
    std::string       str = "alltoall:64-256:cuda:10";
    ucc_coll_score_t *score;
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_CUDA,
                          RLIST({RANGE(64, 256, 10)})));

    EXPECT_NE(UCC_OK,
              check_range(score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(64, 256, 10)})));
    ucc_coll_score_free(score);

    str = "10-20:scatter:host:99";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_SCATTER, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(10, 20, 99)})));
    ucc_coll_score_free(score);
}

UCC_TEST_F(test_score_str, check_range_multiple)
{
    std::string       str = "alltoall:1k-2k,64-256,4096-5000:cuda:10";
    ucc_coll_score_t *score;
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_CUDA,
                          RLIST({RANGE(64, 256, 10), RANGE(1024,2048,10), RANGE(4096,5000,10)})));
    ucc_coll_score_free(score);

    str = "alltoall,barrier:1k-4K,64-256:cuda:10#20:99-12M:bcast";
    EXPECT_EQ(UCC_OK, ucc_coll_score_alloc_from_str(str.c_str(), &score, 0,
                                                    NULL, NULL, NULL));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_CUDA,
                          RLIST({RANGE(64, 256, 10), RANGE(1024,4096,10)})));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BARRIER, UCC_MEMORY_TYPE_CUDA,
                          RLIST({RANGE(64, 256, 10), RANGE(1024,4096,10)})));
    EXPECT_EQ(UCC_OK,
              check_range(score, UCC_COLL_TYPE_BCAST, UCC_MEMORY_TYPE_HOST,
                          RLIST({RANGE(99, 12*1024*1024, 20)})));
    ucc_coll_score_free(score);
}
