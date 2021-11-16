/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_COLL_H_
#define UCC_TL_SHM_COLL_H_

#include "tl_shm.h"

#define MAX_INLINE(_CS, _data, _inline)                                       \
    do {                                                                      \
        _inline = _CS - ucc_offsetof(ucc_tl_shm_ctrl_t, _data);               \
    } while (0)

typedef struct ucc_tl_shm_task {
	ucc_coll_task_t   super;
    union {
        struct {
            ucc_tl_shm_seg_t  *seg;
            ucc_tl_shm_tree_t *tree;
            uint32_t           seq_num;
        };
    };
} ucc_tl_shm_task_t;

static inline ucc_tl_shm_ctrl_t *ucc_tl_shm_get_ctrl(ucc_tl_shm_seg_t *seg,
                                 ucc_tl_shm_team_t *team,
                                 ucc_rank_t rank /* rank within a TL team */)
{
    return PTR_OFFSET(seg->ctrl, ucc_ep_map_eval(team->ctrl_map, rank));
}

static inline ucc_tl_shm_ctrl_t *ucc_tl_shm_get_data(ucc_tl_shm_seg_t *seg,
                                 ucc_tl_shm_team_t *team,
                                 ucc_rank_t rank) /* rank withing a TL team */
{
	size_t data_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.data_size;
    return PTR_OFFSET(seg->data, data_size * rank);
}

static inline void copy_to_children(ucc_tl_shm_seg_t *seg,
                                    ucc_tl_shm_team_t *team,
                                    ucc_kn_tree_t *tree,
                                    int is_inline,
                                    void *src, size_t data_size)
{
    ucc_tl_shm_ctrl_t *ctrl;
    uint32_t seq_num = task->seq_num;

    for (int i =0 ;i < tree->n_children; i++) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        void *dst = is_inline ? ctrl->data : ucc_tl_shm_get_data(seg, team, tree->children[i]);
        memcpy(dst, src, data_size);
        SHMSEG_WMB();
        ctrl->pi = seq_num;
    }
}

static inline void signal_to_children(ucc_tl_shm_seg_t *seg,
                                      ucc_tl_shm_team_t *team,
                                      uint32_t seq_num,
                                      ucc_kn_tree_t *tree)
{
    ucc_tl_shm_ctrl_t *ctrl;
    for (int i =0 ;i < tree->n_children; i++) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        ctrl->pi = seq_num;
    }
}

static inline int ucc_tl_shm_cache_tree_lookup(ucc_tl_shm_team_t *team, ucc_rank_t base_radix, ucc_rank_t top_radix,
                                 ucc_rank_t root, ucc_coll_type_t coll_type, ucc_tl_shm_tree_t **tree) {
    ucc_tl_shm_tree_cache_t tree_cache = team->tree_cache;
    for (int i = 0; i < tree_cache.size; i++) {
    	ucc_tl_shm_tree_cache_keys_t *keys = PTR_OFFSET(tree_cache,
                                     sizeof(ucc_tl_shm_tree_cache_keys_t) * i);
        if (keys->base_radix == base_radix && keys->top_radix == top_radix && keys->root == root && keys->coll_type == coll_type) {
        	*tree = PTR_OFFSET(tree_cache.trees,
                               (sizeof(ucc_tl_shm_tree_t) * self->size * i);
        	return 1;
        }
    }
    return 0;
}

static inline void ucc_tl_shm_cache_tree(ucc_tl_shm_team_t *team, ucc_rank_t base_radix, ucc_rank_t top_radix,
                           ucc_rank_t root, ucc_coll_type_t coll_type, ucc_tl_shm_tree_t *tree) {
    ucc_tl_shm_tree_cache_t *tree_cache = team->tree_cache;
    if (tree_cache->size < UCC_TL_SHM_TEAM_LIB(team)->cfg.max_trees_cached) { //what to do if not?
        ucc_tl_shm_tree_cache_keys_t *cache_keys = PTR_OFFSET(tree_cache->keys, sizeof(ucc_tl_shm_tree_cache_keys_t) * tree_cache->size);
        ucc_tl_shm_tree_t *cache_tree = PTR_OFFSET(tree_cache->trees, sizeof(ucc_tl_shm_tree_t) * tree_cache->size);

        *cache_tree = tree;
        cache_keys->base_radix = base_radix;
        cache_keys->top_radix = top_radix;
        cache_keys->root = root;
        cache_keys->coll_type = coll_type;
        tree_cache->size++;
    }
}

#endif
