/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm_knomial_pattern.h"

void ucc_tl_shm_kn_tree_init(ucc_rank_t size, /* group size */
                                    ucc_rank_t root, /* root of bcast*/
                                    ucc_rank_t rank, /* calling rank */
                                    ucc_rank_t radix, /* kn radix */
                                    ucc_coll_type_t coll_type, /* bcast/reduce */
                                    ucc_kn_tree_t *tree_p /* output tree */)
{
    int pos, i;
    ucc_rank_t peer, vpeer, vparent;
    ucc_rank_t *peer_ptr;
    ucc_rank_t dist = 1;
    ucc_rank_t vrank = (rank - root + size) % size;
    int n_children = 0, calc_parent = 0;

    printf("in tree init, rank = %d\n", rank);

    if (coll_type == UCC_COLL_TYPE_BCAST) {
        while (dist < size) {
    	    dist *= radix;
        }
    }

    while (CONDITION(dist, size, coll_type)) {
        if (vrank % dist == 0) {
//        	printf("rank %d iniside first if\n", rank);
            pos = (vrank / dist) % radix;
            if (pos == 0) {
//            	printf("rank %d iniside pos == 0\n", rank);
                i = coll_type == UCC_COLL_TYPE_BCAST ? radix - 1 : 1;
                while (CONDITION(i, radix, coll_type)) {
                    vpeer = vrank + i * dist;
                    if (vpeer < size) {
                        peer = (vpeer + root) % size;
                        peer_ptr = (ucc_rank_t *) PTR_OFFSET(&tree_p->children[0], sizeof(ucc_rank_t) * n_children);
                        *peer_ptr = peer;
                        n_children++;
                    } else if (coll_type == UCC_COLL_TYPE_REDUCE) {
                    	break;
                    }
                    i = coll_type == UCC_COLL_TYPE_BCAST ? i - 1 : i + 1;
        	    }
            } else if (!calc_parent) {
//            	printf("rank %d iniside pos else\n", rank);
            	vparent = vrank - pos * dist;
                tree_p->parent = (vparent + root) % size;
            	calc_parent = 1;
        	}
        }
        dist = coll_type == UCC_COLL_TYPE_BCAST ? dist / radix : dist * radix;
    }
    if (rank == root) {
        tree_p->parent = UCC_RANK_INVALID;
        printf("inside root parent %d\n", UCC_RANK_INVALID);
    }
    if (n_children == 0) {
    	*tree_p->children = UCC_RANK_INVALID;
    }
    tree_p->n_children = n_children;
    printf("radix = %d, root = %d, rank = %d, parent = %d, n_children = %d \n", radix, root, rank, tree_p->parent, tree_p->n_children);
    for (i = 0; i < n_children; i ++) {
    	printf("child num %d = %d \n", i+1, tree_p->children[i]);
    }
}
