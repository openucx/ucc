/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_SCORE_H
#define UCC_TEST_SCORE_H
extern "C" {
#include "coll_score/ucc_coll_score.h"
}
#include <common/test.h>
#include <string>
#include <vector>

typedef std::tuple<size_t, size_t, ucc_score_t> range_t;
typedef std::tuple<ucc_score_t, uint64_t>       fallback_t;

#define RANGE(_start, _end, _score) std::make_tuple(_start, _end, _score)
#define RLIST(...) std::vector<range_t>(__VA_ARGS__)

#define FB(_score, _init) std::make_tuple(_score, _init)
#define FB_LIST(...) std::vector<fallback_t>(__VA_ARGS__)
#define FB_LLIST(...) std::vector<std::vector<fallback_t>>(__VA_ARGS__)

#define FIRST_RANGE(_score, _ct, _mt)                                          \
    ({                                                                         \
        ucc_list_link_t *l =                                                   \
            _score->scores[ucc_ilog2(UCC_COLL_TYPE_##_ct)]                     \
                          [UCC_MEMORY_TYPE_##_mt]                              \
                              .next;                                           \
        ucc_msg_range_t *range =                                               \
            ucc_container_of(l, ucc_msg_range_t, super.list_elem);             \
        range;                                                                 \
    })

class test_score : public ucc::test {
public:
    ucc_status_t check_range(ucc_coll_score_t *score, ucc_coll_type_t c,
                             ucc_memory_type_t m, std::vector<range_t> check);
    ucc_status_t check_fallback(ucc_coll_score_t *score, ucc_coll_type_t c,
                                ucc_memory_type_t                    m,
                                std::vector<std::vector<fallback_t>> check);
    void         init_score(ucc_coll_score_t *score, std::vector<range_t> v,
                            ucc_coll_type_t c, uint64_t init_fn = 0, uint64_t team = 0);
};

#endif
