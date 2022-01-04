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
            int                tree_in_cache;
            int                progress_in_top_tree;
//            int                base_tree_only;
            uint32_t           seq_num;
//            uint32_t           progress_alg
//            ucc_rank_t         base_radix;
            ucc_rank_t         cur_child;
        };
    };
} ucc_tl_shm_task_t;


ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

int ucc_tl_shm_cache_tree_lookup(ucc_tl_shm_team_t *team,
                                 ucc_rank_t base_radix, ucc_rank_t top_radix,
                                 ucc_rank_t root, ucc_coll_type_t coll_type,
                                 ucc_tl_shm_tree_t **tree);

int ucc_tl_shm_cache_tree(ucc_tl_shm_team_t *team, ucc_rank_t base_radix,
                          ucc_rank_t top_radix, ucc_rank_t root,
                          ucc_coll_type_t coll_type, ucc_tl_shm_tree_t *tree);

ucc_status_t ucc_tl_shm_tree_init(ucc_tl_shm_team_t *team, ucc_rank_t root,
                                  ucc_rank_t base_radix, ucc_rank_t top_radix,
                                  int *tree_in_cache,
                                  ucc_coll_type_t coll_type,
                                  ucc_tl_shm_tree_t **tree_p);

//void ucc_tl_shm_set_coll_perf_params(ucc_tl_shm_task_t *task, ucc_coll_type_t coll_type);


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

static inline ucc_status_t ucc_tl_shm_seg_ready(ucc_tl_shm_seg_t  *seg)
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
    void              *dst;
    int                i;

    for (i = 0; i < tree->n_children; i++) {
//    for (i = tree->n_children - 1 ;i >= 0; i--) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        dst = is_inline ? ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                           tree->children[i]);
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
    int                i;
    for (i = 0; i < tree->n_children; i++) {
//    for (i = tree->n_children - 1; i >= 0; i--) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        ctrl->pi = seq_num;
    }
}

#endif
