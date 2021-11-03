/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
extern "C" {
#include "utils/ucc_coll_utils.h"
}
#include <common/test.h>
#include <string>
#include <vector>

class EpMap {
    ucc_ep_map_t map;
public:
    ucc_rank_t *array;
    EpMap(ucc_ep_map_t _map) {
        array = NULL;
        map   = _map;
    };
    EpMap(uint64_t full) {
        map.type   = UCC_EP_MAP_FULL;
        map.ep_num = full;
        array      = NULL;
    };
    EpMap(uint64_t start, int64_t stride, uint64_t num, uint64_t full) {
        map.type           = ((num == full)  && (stride == 1)) ?
            UCC_EP_MAP_FULL : UCC_EP_MAP_STRIDED;
        map.ep_num         = num;
        map.strided.start  = start;
        map.strided.stride = stride;
        array              = NULL;
    };
    EpMap(const std::vector<ucc_rank_t> &ranks, uint64_t full,
          int need_free = 0) {
        array = (ucc_rank_t*)malloc(sizeof(ucc_rank_t) * ranks.size());
        memcpy(array, ranks.data(), ranks.size() * sizeof(ucc_rank_t));
        map = ucc_ep_map_from_array(&array, ranks.size(),
                                    full, need_free);
    }
    ~EpMap() {
        if (array) {
            free(array);
        }
    }
    friend bool operator==(const EpMap &lhs, const EpMap &rhs) {
        if ((lhs.map.type != rhs.map.type) ||
            (lhs.map.ep_num != rhs.map.ep_num)) {
            return false;
        }
        switch(lhs.map.type) {
        case UCC_EP_MAP_FULL:
            return true;
        case UCC_EP_MAP_STRIDED:
            return (lhs.map.strided.start == rhs.map.strided.start) &&
                (lhs.map.strided.stride == rhs.map.strided.stride);
        default:
            break;
        }
        return false;
    }
};

class test_ep_map : public ucc::test {};

UCC_TEST_F(test_ep_map, from_array)
{
    // Full contiguous map
    EXPECT_EQ(EpMap({1,2,3,4,5,6,7,8,9,10}, 10), EpMap(10));

    // Strided contiguous map
    EXPECT_EQ(EpMap({1,31,61,91,121}, 150), EpMap(1, 30, 5, 150));

    // Strided negative
    EXPECT_EQ(EpMap({100,90,80,70,60}, 150), EpMap(100, -10, 5, 150));
}

UCC_TEST_F(test_ep_map, from_array_free)
{
    /* strided pattern not found - array is not released */
    EXPECT_NE((void*)NULL, EpMap({1, 5, 6, 8, 11}, 10, 1).array);

    /* strided pattern found - array is released */
    EXPECT_EQ((void*)NULL, EpMap({2, 4, 6, 8, 10}, 10, 1).array);

    /* FULL pattern found - array is released */
    EXPECT_EQ((void*)NULL, EpMap({1, 2, 3, 4, 5}, 5, 1).array);
}
