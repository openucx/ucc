/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "utils/ucc_coll_utils.h"
}
#include <common/test.h>
#include <string>
#include <vector>

class EpMap {
public:
  ucc_ep_map_t map;
  ucc_rank_t * array;
  EpMap(){};
  EpMap(ucc_ep_map_t _map)
  {
      array = NULL;
      map   = _map;
    };
    EpMap(uint64_t full, bool reverse = false)
    {
        array      = NULL;
        if (reverse) {
            map = ucc_ep_map_create_reverse(full);
        } else {
            map.type   = UCC_EP_MAP_FULL;
            map.ep_num = full;
        }
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

UCC_TEST_F(test_ep_map, reverse)
{
    const int    size = 10;
    ucc_ep_map_t map  = ucc_ep_map_create_reverse(size);

    for (int i = 0; i < size; i++) {
        EXPECT_EQ(size - 1 - i, ucc_ep_map_eval(map, i));
    }
}

UCC_TEST_F(test_ep_map, nested)
{
    auto map1 = EpMap(100); //full map, size 100
    auto map2 = EpMap(0, 2, 50, 100); // submap even only
    auto map3 = EpMap(1, 2, 25, 50); // submap odd only from 50
    ucc_ep_map_t nested1, nested2;

    EXPECT_EQ(UCC_OK, ucc_ep_map_create_nested(&map1.map, &map2.map, &nested1));
    EXPECT_EQ(50, nested1.ep_num);
    for (int i = 0; i < nested1.ep_num; i++) {
        EXPECT_EQ(0 + i * 2, ucc_ep_map_eval(nested1, i));
    }

    EXPECT_EQ(UCC_OK, ucc_ep_map_create_nested(&nested1, &map3.map, &nested2));
    EXPECT_EQ(25, nested2.ep_num);
    for (int i = 0; i < nested2.ep_num; i++) {
        EXPECT_EQ(2 + i * 4, ucc_ep_map_eval(nested2, i));
    }

    ucc_ep_map_destroy_nested(&nested1);
    ucc_ep_map_destroy_nested(&nested2);
}

class test_ep_map_inv : public test_ep_map {
  public:
    void check_inv(EpMap map)
    {
        ucc_ep_map_t inv;
        EXPECT_EQ(UCC_OK, ucc_ep_map_create_inverse(map.map, &inv));
        for (int i = 0; i < map.map.ep_num; i++) {
            EXPECT_EQ(i, ucc_ep_map_eval(inv, ucc_ep_map_eval(map.map, i)));
        }
        ucc_ep_map_destroy(&inv);
    };
};

UCC_TEST_F(test_ep_map_inv, contig)
{
    /* reverse of FULL */
    check_inv(EpMap(10));

    /* reverse of INVERSE */
    check_inv(EpMap(10, true));
}

UCC_TEST_F(test_ep_map_inv, strided)
{
    /* stride positive */
    check_inv(EpMap(1, 30, 5, 150));

    /* stride negative */
    check_inv(EpMap(100, -10, 5, 150));
}

UCC_TEST_F(test_ep_map_inv, random)
{
    check_inv(EpMap({4, 0, 1, 2, 3}, 5));
}
