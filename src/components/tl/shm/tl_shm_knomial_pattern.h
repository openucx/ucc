/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TL_SHM_KNOMIAL_PATTERN_H_
#define TL_SHM_KNOMIAL_PATTERN_H_

#define CONDITION(_a, _b, _coll) _coll == UCC_COLL_TYPE_BCAST ? _a >= 1 : _a < _b // coll_type can be bcast or reduce
#define GET_MAX_RADIX(_r1, _r2) _r1 > _r2 ? _r1 : _r2

typedef struct ucc_kn_tree {
    ucc_rank_t parent;
    ucc_rank_t n_children;
    ucc_rank_t children[1];
} ucc_kn_tree_t;

/* returns upper bound of rank's tree size (parent + children) */
static inline size_t ucc_tl_shm_kn_tree_size(size_t size, ucc_rank_t radix)
{
	size_t log_size = 1;
	while (size > radix) {
		size /= radix;
		log_size++;
	}
	return log_size * (radix - 1) + 1;
}

static void ucc_tl_shm_kn_tree_init(ucc_rank_t size, /* group size */
                                    ucc_rank_t root, /* root of bcast*/
                                    ucc_rank_t rank, /* calling rank */
                                    ucc_rank_t radix, /* kn radix */
                                    ucc_coll_type_t coll_type, /* bcast/reduce */
                                    ucc_kn_tree_t *tree_p /* output tree */)
{
    int pos, i;
    ucc_rank_t peer, vpeer, vparent;
    ucc_rank_t dist = 1;
    ucc_rank_t vrank = (rank - root + size) % size;
    int n_children = 0, calc_parent = 0;

    if (coll_type == UCC_COLL_TYPE_BCAST) {
        while (dist < size) {
    	    dist *= radix;
        }
    }

    tree_p->children = NULL;
    while (CONDITION(dist, size, coll_type)) {
        if (vrank % dist == 0) {
            pos = (vrank / dist) % radix;
            if (pos == 0) {
                i = coll_type == UCC_COLL_TYPE_BCAST ? radix - 1 : 1;
                while (CONDITION(i, radix, coll_type)) {
                    vpeer = vrank + i * dist;
                    if (vpeer < size) {
                        peer = (vpeer + root) % team_size;
                        *tree_p->children = peer;
                        tree_p->children = PTR_OFFSET(tree_p->children, sizeof(ucc_rank_t));
                        n_children++;
                    } else if (coll_type == UCC_COLL_TYPE_REDUCE) {
                    	break;
                    }
                    i = coll_type == UCC_COLL_TYPE_BCAST ? i - 1 : i + 1;
        	    }
            } else if (!calc_parent) {
            	vparent = vrank - pos * dist;
                tree_p->parent = (vparent + root) % team_size;
                calc_parent = 1;
        	}
        }
        dist = coll_type == UCC_COLL_TYPE_BCAST ? dist / radix : dist * radix;
    }
    tree_p->n_children = n_children;
}

#endif
