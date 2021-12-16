/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "tl_shm_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"

ucc_status_t ucc_tl_shm_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                       ucc_coll_task_t *coll_task)
{
    return UCC_OK;
}

static ucc_status_t ucc_tl_shm_coll_finalize(ucc_coll_task_t *coll_task)
{
	ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	if (!task->tree_in_cache) {
	    ucc_free(task->tree->base_tree);
	    ucc_free(task->tree->top_tree);
	    ucc_free(task->tree);
	}
	UCC_TL_SHM_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h)
{
    ucc_status_t          status = UCC_OK;
    ucc_tl_shm_context_t *ctx    = ucc_derived_of(team->context, ucc_tl_shm_context_t);
    ucc_tl_shm_task_t    *task   = ucc_mpool_get(&ctx->req_mp);
    UCC_TL_SHM_PROFILE_REQUEST_NEW(task, "tl_shm_task", 0);
	ucc_coll_task_init(&task->super, coll_args, team);

	task->super.finalize = ucc_tl_shm_coll_finalize;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_shm_bcast_init(task);
        break;
    case UCC_COLL_TYPE_REDUCE:
        status = ucc_tl_shm_reduce_init(task);
        break;
    default:
    	ucc_free(task);
    	return UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(status != UCC_OK)) {
    	ucc_free(task);
    	tl_error(team->context->lib, "bcast init failed");
        return status;
    }
    tl_trace(team->context->lib, "init coll req %p", task);

	*task_h = &task->super;
    return UCC_OK;
}

int ucc_tl_shm_cache_tree_lookup(ucc_tl_shm_team_t *team,
                                 ucc_rank_t base_radix, ucc_rank_t top_radix,
                                 ucc_rank_t root, ucc_coll_type_t coll_type,
                                 ucc_tl_shm_tree_t **tree) {
    ucc_tl_shm_tree_cache_elems_t *elems = team->tree_cache->elems;

    for (int i = 0; i < team->tree_cache->size; i++) {
        if (elems[i].keys.base_radix == base_radix &&
            elems[i].keys.top_radix == top_radix &&
            elems[i].keys.root == root &&
            elems[i].keys.coll_type == coll_type) {
            *tree = elems[i].tree;
            return 1;
        }
    }
    return 0;
}

int ucc_tl_shm_cache_tree(ucc_tl_shm_team_t *team, ucc_rank_t base_radix,
                          ucc_rank_t top_radix, ucc_rank_t root,
                          ucc_coll_type_t coll_type, ucc_tl_shm_tree_t *tree) {
    size_t size = team->tree_cache->size;
    ucc_tl_shm_tree_cache_elems_t *elem = &team->tree_cache->elems[size];

    if (size < UCC_TL_SHM_TEAM_LIB(team)->cfg.max_trees_cached) {
        elem->tree = tree;
        elem->keys.base_radix = base_radix;
        elem->keys.top_radix = top_radix;
        elem->keys.root = root;
        elem->keys.coll_type = coll_type;
        team->tree_cache->size++;
        return 1;
    }
    return 0;
}

ucc_status_t ucc_tl_shm_tree_init(ucc_tl_shm_team_t *team,
                                  ucc_rank_t root,
                                  ucc_rank_t base_radix,
                                  ucc_rank_t top_radix,
                                  int *tree_in_cache,
                                  ucc_coll_type_t coll_type,
                                  ucc_tl_shm_tree_t **tree_p)
{
    ucc_kn_tree_t     *base_tree, *top_tree;
    ucc_tl_shm_tree_t *shm_tree;
    ucc_rank_t         tree_root, rank, local_rank, leader_rank, leader_group_id;
    size_t             top_tree_size, base_tree_size;
    int i;

    ucc_rank_t  team_rank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t  group_size   = team->base_groups[team->my_group_id].group_size;
    ucc_rank_t  leaders_size = team->leaders_group->group_size;
    ucc_rank_t  root_group   = ucc_ep_map_eval(team->rank_group_id_map, root);
    ucc_sbgp_t *sbgp         = &team->base_groups[team->my_group_id];

    if (ucc_tl_shm_cache_tree_lookup(team, base_radix, top_radix, root,
                                     coll_type, tree_p) == 1) {
    	*tree_in_cache = 1;
        return UCC_OK;
    }
    /* Pool is initialized using UCC_KN_TREE_SIZE macro memory estimation, using
       base_group[my_group_id]->size and max supported radix (maybe up to group size as well */

    shm_tree = (ucc_tl_shm_tree_t *) ucc_malloc(sizeof(ucc_kn_tree_t *) * 2);
    top_tree_size = ucc_tl_shm_kn_tree_size(leaders_size, top_radix);
    base_tree_size = ucc_tl_shm_kn_tree_size(group_size, base_radix);
    base_tree = (ucc_kn_tree_t *) ucc_malloc(sizeof(ucc_rank_t) *
                                             (base_tree_size + 2));

    if (!base_tree) {
        return UCC_ERR_NO_MEMORY;
    }

    top_tree = (ucc_kn_tree_t *) ucc_malloc(sizeof(ucc_rank_t) *
                                            (top_tree_size + 2));

    if (!top_tree) {
    	ucc_free(base_tree);
        return UCC_ERR_NO_MEMORY;
    }

    shm_tree->base_tree = NULL;
    shm_tree->top_tree = NULL;
    for (i = 0; i < sbgp->group_size; i++) {
    	if (ucc_ep_map_eval(sbgp->map, i) == team_rank) {
    		local_rank = i;
    		break;
    	}
    }
    ucc_assert(i < sbgp->group_size);

    if (group_size > 1) {
        rank = local_rank;
        if (team->my_group_id == root_group) {
            tree_root = ucc_ep_map_eval(team->group_rank_map, root);
        } else {
            tree_root = 0;
        }
        ucc_tl_shm_kn_tree_init(group_size, tree_root, rank, base_radix,
                                coll_type, base_tree);
        shm_tree->base_tree = base_tree;
        /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
        ucc_tl_shm_tree_to_team_ranks(base_tree,
                                      team->base_groups[team->my_group_id].map);
    }
    if (leaders_size > 1) {
        if (team_rank == root ||
            (root_group != team->my_group_id &&
             UCC_SBGP_ENABLED == team->leaders_group->status)) {
            /* short cut if root is part of leaders SBGP
               Loop below is up to number of sockets/numas, ie small and fast*/
            tree_root = UCC_RANK_INVALID;
            for (i = 0; i < team->leaders_group->group_size; i++) {
                if (ucc_ep_map_eval(team->leaders_group->map, i) == root) {
                    tree_root = i;
                    break;
                }
            }
            rank = team->leaders_group->group_rank;
            if (tree_root != UCC_RANK_INVALID) {
                /* Root is part of leaders groop */
            	ucc_tl_shm_kn_tree_init(leaders_size, tree_root, rank,
                                        top_radix, coll_type,
                                        top_tree);
                /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
                ucc_tl_shm_tree_to_team_ranks(top_tree,
                                              team->leaders_group->map);
            } else {
                /* Build tmp ep_map for leaders + root set
                  The Leader Rank on the same base group with actual root will be replaced in the tree
                  by the root itself to save 1 extra copy in SM */
                ucc_rank_t ranks[leaders_size]; //Can be allocated on stack
                for (i = 0; i < leaders_size; i++) {
                    leader_rank = ucc_ep_map_eval(team->leaders_group->map, i);
                    leader_group_id = ucc_ep_map_eval(team->rank_group_id_map,
                                                      leader_rank);
                    if (leader_group_id == root_group) {
                        tree_root = i;
                        ranks[i] = root;
                        if (team_rank == root) {
                            rank = i;
                        }
                    } else {
                        ranks[i] = leader_rank;
                    }
                }
                ucc_tl_shm_kn_tree_init(leaders_size, tree_root, rank,
                                        coll_type, top_radix,
                                        top_tree);
                /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
                ucc_ep_map_t map = {
                    .type  = UCC_EP_MAP_ARRAY,
                    .array.map = ranks,
                    .array.elem_size = sizeof(ucc_rank_t)
                };
                ucc_tl_shm_tree_to_team_ranks(top_tree, map);
            }
            shm_tree->top_tree = top_tree;
        }
    }
    *tree_in_cache = ucc_tl_shm_cache_tree(team, base_radix, top_radix, root,
                                           coll_type, shm_tree);
    *tree_p = shm_tree;
    return UCC_OK;
}
