/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "common/gtest.h"
#include "components/tl/ucp/alltoall/alltoall_pairwise_schedule.h"

#include <vector>

TEST(alltoall_pairwise_schedule, block_mapping)
{
    ucc_ep_map_t     node_maps[2] = {};
    const ucc_rank_t expected[] = {0, 4, 1, 5, 2, 6, 3, 7};
    ucc_rank_t       rank_order[8], rank_labels[8];

    for (ucc_rank_t node = 0; node < 2; node++) {
        node_maps[node].type            = UCC_EP_MAP_STRIDED;
        node_maps[node].ep_num          = 4;
        node_maps[node].strided.start   = node * 4;
        node_maps[node].strided.stride  = 1;
    }
    EXPECT_EQ(UCC_OK, ucc_tl_ucp_alltoall_node_interleaved_build_map(
                          node_maps, 8, 2, 4, rank_order, rank_labels));
    for (ucc_rank_t i = 0; i < 8; i++) {
        EXPECT_EQ(expected[i], rank_order[i]);
        EXPECT_EQ(i, rank_labels[rank_order[i]]);
    }
}

TEST(alltoall_pairwise_schedule, reordered_team_mapping)
{
    const ucc_rank_t node_ranks[] = {4, 1, 5, 0, 3, 7, 2, 6};
    const ucc_rank_t expected[]   = {4, 5, 3, 2, 1, 0, 7, 6};
    ucc_ep_map_t     node_maps[4] = {};
    ucc_rank_t       rank_order[8], rank_labels[8];

    for (ucc_rank_t node = 0; node < 4; node++) {
        node_maps[node].type            = UCC_EP_MAP_ARRAY;
        node_maps[node].ep_num          = 2;
        node_maps[node].array.map       = (void *)&node_ranks[node * 2];
        node_maps[node].array.elem_size = sizeof(ucc_rank_t);
    }
    EXPECT_EQ(UCC_OK, ucc_tl_ucp_alltoall_node_interleaved_build_map(
                          node_maps, 8, 4, 2, rank_order, rank_labels));
    for (ucc_rank_t i = 0; i < 8; i++) {
        EXPECT_EQ(expected[i], rank_order[i]);
        EXPECT_EQ(i, rank_labels[rank_order[i]]);
    }
}

TEST(alltoall_pairwise_schedule, rejects_invalid_geometry_and_maps)
{
    const ucc_rank_t valid[]        = {0, 1, 2, 3};
    const ucc_rank_t duplicate[]    = {0, 1, 1, 3};
    const ucc_rank_t out_of_range[] = {0, 1, 2, 4};
    ucc_ep_map_t     maps[2] = {};
    ucc_rank_t       rank_order[4], rank_labels[4];

    for (ucc_rank_t node = 0; node < 2; node++) {
        maps[node].type            = UCC_EP_MAP_ARRAY;
        maps[node].ep_num          = 2;
        maps[node].array.elem_size = sizeof(ucc_rank_t);
    }
    maps[0].array.map = (void *)&valid[0];
    maps[1].array.map = (void *)&valid[2];
    EXPECT_EQ(UCC_ERR_INVALID_PARAM,
              ucc_tl_ucp_alltoall_node_interleaved_build_map(
                  maps, 4, 1, 4, rank_order, rank_labels));
    EXPECT_EQ(UCC_ERR_INVALID_PARAM,
              ucc_tl_ucp_alltoall_node_interleaved_build_map(
                  maps, 4, 2, 3, rank_order, rank_labels));
    maps[1].ep_num = 1;
    EXPECT_EQ(UCC_ERR_INVALID_PARAM,
              ucc_tl_ucp_alltoall_node_interleaved_build_map(
                  maps, 4, 2, 2, rank_order, rank_labels));
    maps[1].ep_num = 2;
    maps[0].array.map = (void *)&duplicate[0];
    maps[1].array.map = (void *)&duplicate[2];
    EXPECT_EQ(UCC_ERR_INVALID_PARAM,
              ucc_tl_ucp_alltoall_node_interleaved_build_map(
                  maps, 4, 2, 2, rank_order, rank_labels));
    maps[0].array.map = (void *)&out_of_range[0];
    maps[1].array.map = (void *)&out_of_range[2];
    EXPECT_EQ(UCC_ERR_INVALID_PARAM,
              ucc_tl_ucp_alltoall_node_interleaved_build_map(
                  maps, 4, 2, 2, rank_order, rank_labels));
}

TEST(alltoall_pairwise_schedule, every_peer_and_reciprocal_receive)
{
    for (ucc_rank_t ppn : {2, 4, 8}) {
        for (ucc_rank_t nnodes : {2, 3, 8, 16}) {
            ucc_rank_t size = ppn * nnodes;
            std::vector<ucc_rank_t> node_ranks(size);
            std::vector<ucc_rank_t> rank_order(size);
            std::vector<ucc_rank_t> rank_labels(size);
            std::vector<ucc_ep_map_t> node_maps(nnodes);

            for (ucc_rank_t i = 0; i < size; i++) {
                node_ranks[i] = size - i - 1;
            }
            for (ucc_rank_t node = 0; node < nnodes; node++) {
                node_maps[node].type            = UCC_EP_MAP_ARRAY;
                node_maps[node].ep_num          = ppn;
                node_maps[node].array.map       = &node_ranks[node * ppn];
                node_maps[node].array.elem_size = sizeof(ucc_rank_t);
            }
            ASSERT_EQ(UCC_OK, ucc_tl_ucp_alltoall_node_interleaved_build_map(
                                  node_maps.data(), size, nnodes, ppn,
                                  rank_order.data(), rank_labels.data()));

            for (ucc_rank_t rank = 0; rank < size; rank++) {
                std::vector<int> seen(size, 0);
                for (ucc_rank_t step = 0; step < size; step++) {
                    ucc_rank_t send_peer =
                        ucc_tl_ucp_alltoall_node_interleaved_peer(
                            rank_order.data(), rank_labels.data(), rank, size,
                            step, 1);
                    ucc_rank_t reciprocal =
                        ucc_tl_ucp_alltoall_node_interleaved_peer(
                            rank_order.data(), rank_labels.data(), send_peer,
                            size, step, 0);

                    EXPECT_LT(send_peer, size);
                    EXPECT_EQ(0, seen[send_peer]++);
                    EXPECT_EQ(rank, reciprocal);
                }
            }
        }
    }
}
