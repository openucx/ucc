/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TL_SHM_KNOMIAL_PATTERN_H_
#define TL_SHM_KNOMIAL_PATTERN_H_

#include "tl_shm.h"

/* checks if to build kn tree children setup to fit bcast/fanout or reduce/fanin */
#define CONDITION(_a, _b, _coll)                                           \
    (((_coll) == UCC_COLL_TYPE_BCAST || (_coll) == UCC_COLL_TYPE_FANOUT) ? \
    ((_a) >= 1) : ((_a) < (_b)))

typedef struct ucc_kn_tree {
    ucc_rank_t parent;
    ucc_rank_t n_children;
    ucc_rank_t children[1];
} ucc_kn_tree_t;

/* returns upper bound of rank's tree size (parent + children) */
static inline size_t ucc_tl_shm_kn_tree_size(size_t size, ucc_rank_t radix)
{
    size_t log_size = 1;
    ucc_rank_t fs   = radix;

    while (fs < size) {
        fs *= radix;
        log_size++;
    }

    return sizeof(ucc_kn_tree_t) + sizeof(ucc_rank_t) *
                                   (log_size * (radix - 1) - 1);
}

static inline void ucc_tl_shm_tree_to_team_ranks(ucc_kn_tree_t *tree,
                                                 ucc_ep_map_t   map)
{
    if (tree->parent != UCC_RANK_INVALID) {
        tree->parent = ucc_ep_map_eval(map, tree->parent);
    }
    for (int i=0; i<tree->n_children; i++) {
        tree->children[i] = ucc_ep_map_eval(map, tree->children[i]);
    }
}

void ucc_tl_shm_kn_tree_init(ucc_rank_t size, /* group size */
                             ucc_rank_t root, /* root of bcast*/
                             ucc_rank_t rank, /* calling rank */
                             ucc_rank_t radix, /* kn radix */
                             ucc_coll_type_t coll_type, /* bcast/reduce */
                             ucc_kn_tree_t *tree_p /* output tree */);

#endif
