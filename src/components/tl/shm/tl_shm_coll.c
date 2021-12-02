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
//#include "reduce/reduce.h"

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
	ucc_free(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h)
{
	ucc_status_t status = UCC_OK;
	ucc_tl_shm_task_t *task =
                (ucc_tl_shm_task_t *) ucc_malloc(sizeof(ucc_tl_shm_task_t));
	ucc_coll_task_init(&task->super, coll_args, team);

	task->super.finalize = ucc_tl_shm_coll_finalize;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_shm_bcast_init(task);
        break;
//    case UCC_COLL_TYPE_REDUCE:
//        status = ucc_tl_shm_reduce_init(task);
//        break;
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
