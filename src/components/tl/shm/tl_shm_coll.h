/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_COLL_H_
#define UCC_TL_SHM_COLL_H_

#include "tl_shm.h"

typedef struct ucc_tl_shm_task {
	ucc_coll_task_t   super;
    union {
        struct {
            ucc_tl_shm_seg_t  *seg;
            ucc_tl_shm_tree_t *tree;
            void              *src;
            uint32_t           seq_num;
        };
    };
} ucc_tl_shm_task_t;


ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

static inline ucc_tl_shm_ctrl_t *ucc_tl_shm_get_ctrl(ucc_tl_shm_seg_t *seg,
                                 ucc_tl_shm_team_t *team,
                                 ucc_rank_t rank /* rank within a TL team */)
{
    return PTR_OFFSET(seg->ctrl, ucc_ep_map_eval(team->ctrl_map, rank));
}

static inline void *ucc_tl_shm_get_data(ucc_tl_shm_seg_t *seg,
                                 ucc_tl_shm_team_t *team,
                                 ucc_rank_t rank) /* rank withing a TL team */
{
	size_t data_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.data_size;
    return PTR_OFFSET(seg->data, data_size * rank);
}

static inline ucc_status_t ucc_tl_shm_seg_ready()
{
    return UCC_OK;
}

static inline void ucc_tl_shm_copy_to_children(ucc_tl_shm_seg_t *seg,
                                    ucc_tl_shm_team_t *team,
                                    ucc_kn_tree_t *tree,
                                    uint32_t seq_num,
                                    int is_inline,
                                    void *src,
                                    size_t data_size)
{
    ucc_tl_shm_ctrl_t *ctrl;
    void *dst;

    for (int i =0 ;i < tree->n_children; i++) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        dst = is_inline ? ctrl->data : ucc_tl_shm_get_data(seg, team, tree->children[i]);
        memcpy(dst, src, data_size);
        SHMSEG_WMB();
        ctrl->pi = seq_num;
    }
}

static inline void ucc_tl_shm_signal_to_children(ucc_tl_shm_seg_t *seg,
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

static inline int ucc_tl_shm_cache_tree_lookup(ucc_tl_shm_team_t *team,
                                               ucc_rank_t base_radix,
                                               ucc_rank_t top_radix,
                                               ucc_rank_t root,
                                               ucc_coll_type_t coll_type,
                                               ucc_tl_shm_tree_t **tree) {
    ucc_tl_shm_tree_cache_t *team_cache = team->tree_cache;
    for (int i = 0; i < team_cache->size; i++) {
    	ucc_tl_shm_tree_cache_elems_t *elems = PTR_OFFSET(team_cache->elems,
                                sizeof(ucc_tl_shm_tree_cache_elems_t *) * i);
        if (elems->keys->base_radix == base_radix &&
            elems->keys->top_radix == top_radix &&
            elems->keys->root == root &&
            elems->keys->coll_type == coll_type) {
        	*tree = elems->tree;
        	return 1;
        }
    }
    return 0;
}

static inline void ucc_tl_shm_cache_tree(ucc_tl_shm_team_t *team,
                                         ucc_rank_t base_radix,
                                         ucc_rank_t top_radix,
                                         ucc_rank_t root,
                                         ucc_coll_type_t coll_type,
                                         ucc_tl_shm_tree_t *tree) {
    ucc_tl_shm_tree_cache_t *team_cache = team->tree_cache;
    ucc_tl_shm_tree_cache_elems_t **elem_ptr;
    if (team_cache->size < UCC_TL_SHM_TEAM_LIB(team)->cfg.max_trees_cached) {
        ucc_tl_shm_tree_cache_elems_t *cache_elems =
            (ucc_tl_shm_tree_cache_elems_t *)
            ucc_malloc(sizeof(ucc_tl_shm_tree_cache_keys_t) +
            sizeof(ucc_tl_shm_tree_t *));

        cache_elems->tree = tree;
        cache_elems->keys->base_radix = base_radix;
        cache_elems->keys->top_radix = top_radix;
        cache_elems->keys->root = root;
        cache_elems->keys->coll_type = coll_type;
        elem_ptr = (ucc_tl_shm_tree_cache_elems_t **)
                   PTR_OFFSET(&team_cache->elems[0],
                   team_cache->size * sizeof(ucc_tl_shm_tree_cache_elems_t *));
        *elem_ptr = cache_elems;
        team_cache->size++;
    } else {
    	ucc_free(tree);
    }
}

#endif
