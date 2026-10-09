/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_PAIRWISE_SCHEDULE_H_
#define ALLTOALL_PAIRWISE_SCHEDULE_H_

#include "utils/ucc_coll_utils.h"

/*
 * Convert node endpoint maps into a node-interleaved ring order.
 *
 * node_maps contains one team-rank map per node, with the same number of ranks
 * on every node. rank_order maps a virtual ring label to a team rank, while
 * rank_labels provides the inverse mapping. UCC may optimize a node's backing
 * rank array into a strided map, so callers must evaluate the generic map
 * rather than depend on a materialized rank_map.
 */
static inline ucc_status_t
ucc_tl_ucp_alltoall_node_interleaved_build_map(
    const ucc_ep_map_t *node_maps, ucc_rank_t size, ucc_rank_t nnodes,
    ucc_rank_t ppn, ucc_rank_t *rank_order, ucc_rank_t *rank_labels)
{
    ucc_rank_t label, local_rank, node, rank;

    if (!node_maps || !rank_order || !rank_labels || nnodes <= 1 || ppn <= 1 ||
        size / nnodes != ppn || size % nnodes != 0) {
        return UCC_ERR_INVALID_PARAM;
    }

    for (rank = 0; rank < size; rank++) {
        rank_labels[rank] = UCC_RANK_INVALID;
    }

    for (local_rank = 0; local_rank < ppn; local_rank++) {
        for (node = 0; node < nnodes; node++) {
            if (node_maps[node].ep_num != ppn) {
                return UCC_ERR_INVALID_PARAM;
            }
            rank  = ucc_ep_map_eval(node_maps[node], local_rank);
            label = local_rank * nnodes + node;
            if (rank >= size || rank_labels[rank] != UCC_RANK_INVALID) {
                return UCC_ERR_INVALID_PARAM;
            }
            rank_order[label] = rank;
            rank_labels[rank] = label;
        }
    }

    return UCC_OK;
}

static inline ucc_rank_t
ucc_tl_ucp_alltoall_node_interleaved_peer(const ucc_rank_t *rank_order,
                                          const ucc_rank_t *rank_labels,
                                          ucc_rank_t rank, ucc_rank_t size,
                                          ucc_rank_t step, int is_send)
{
    ucc_rank_t label = rank_labels[rank];
    ucc_rank_t peer_label;

    peer_label = is_send ? (label - step + size) % size :
                           (label + step) % size;
    return rank_order[peer_label];
}

#endif
