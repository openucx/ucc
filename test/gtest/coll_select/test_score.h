/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_SCORE_H
#define UCC_TEST_SCORE_H
extern "C" {
#include "coll_select/coll_select.h"
}
#include <common/test.h>
#include <string>
#include <vector>

typedef std::tuple<size_t, size_t, ucc_score_t> range_t;
#define RANGE(_start, _end, _score) std::make_tuple(_start, _end, _score)
#define RLIST(...) std::vector<range_t>(__VA_ARGS__)

class test_score : public ucc::test {
public:
    ucc_status_t check_range(ucc_coll_score_t *score, ucc_coll_type_t c,
                             ucc_memory_type_t m, std::vector<range_t> check);
    void         init_score(ucc_coll_score_t *score, std::vector<range_t> v,
                            ucc_coll_type_t c);
};

#endif
