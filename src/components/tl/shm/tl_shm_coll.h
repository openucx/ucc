/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHM_COLL_H_
#define UCC_TL_SHM_COLL_H_

#include "tl_shm.h"

typedef struct ucc_tl_shm_task {
    ucc_coll_task_t                 super;
    ucc_tl_shm_seg_t *              seg;
    ucc_tl_shm_tree_t *             tree;
    uint32_t                        seq_num;
    uint32_t                        seg_ready_seq_num;
    int                             stage;
    int                             tree_in_cache;
    int                             base_tree_only;
    int                             first_reduce;
    ucc_tl_shm_bcast_progress_alg_t progress_alg;
    ucc_rank_t                      base_radix;
    ucc_rank_t                      top_radix;
    ucc_rank_t                      cur_child;
} ucc_tl_shm_task_t;

ucc_status_t ucc_tl_shm_coll_finalize(ucc_coll_task_t *coll_task);

static inline ucc_tl_shm_task_t *
ucc_tl_shm_get_task(ucc_base_coll_args_t *coll_args, ucc_tl_shm_team_t *team)
{
    ucc_tl_shm_context_t *ctx =
        ucc_derived_of(team->super.super.context, ucc_tl_shm_context_t);
    ucc_tl_shm_task_t *task = ucc_mpool_get(&ctx->req_mp);

    if (ucc_unlikely(!task)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to allocate task");
        return NULL;
    }

    UCC_TL_SHM_PROFILE_REQUEST_NEW(task, "tl_shm_task", 0);
    ucc_coll_task_init(&task->super, coll_args, &team->super.super);
    task->seq_num        = team->seq_num++;
    task->seg            = &team->segs[task->seq_num % team->n_concurrent];
    task->super.finalize = ucc_tl_shm_coll_finalize;
    task->super.triggered_post = ucc_triggered_post;
    task->base_tree_only       = UCC_TL_SHM_TEAM_LIB(team)->cfg.base_tree_only;
    task->first_reduce         = 1;
    task->cur_child            = 0;
    return task;
}

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *     team,
                                  ucc_coll_task_t **    task);

int ucc_tl_shm_cache_tree_lookup(ucc_tl_shm_team_t *          team,
                                 ucc_tl_shm_tree_cache_key_t *key,
                                 ucc_tl_shm_tree_t **         tree);

int ucc_tl_shm_cache_tree(ucc_tl_shm_team_t *          team,
                          ucc_tl_shm_tree_cache_key_t *key,
                          ucc_tl_shm_tree_t *          tree);

ucc_status_t ucc_tl_shm_tree_init(ucc_tl_shm_team_t *team, ucc_rank_t root,
                                  ucc_rank_t base_radix, ucc_rank_t top_radix,
                                  int *tree_in_cache, ucc_coll_type_t coll_type,
                                  int                 base_tree_only,
                                  ucc_tl_shm_tree_t **tree_p);

static inline ucc_tl_shm_ctrl_t *
ucc_tl_shm_get_ctrl(ucc_tl_shm_seg_t *seg, ucc_tl_shm_team_t *team,
                    ucc_rank_t rank /* rank within a TL team */)
{
    return PTR_OFFSET(seg->ctrl, ucc_ep_map_eval(team->ctrl_map, rank));
}

static inline void *
ucc_tl_shm_get_data(ucc_tl_shm_seg_t *seg, ucc_tl_shm_team_t *team,
                    ucc_rank_t rank) /* rank withing a TL team */
{
    size_t data_size = UCC_TL_SHM_TEAM_LIB(team)->cfg.data_size;

    return PTR_OFFSET(seg->data, data_size * rank);
}

static inline ucc_status_t ucc_tl_shm_bcast_seg_ready(ucc_tl_shm_seg_t *seg,
                                                      uint32_t          seq_num,
                                                      ucc_tl_shm_team_t *team,
                                                      ucc_tl_shm_tree_t *tree)
{
    ucc_tl_shm_ctrl_t *ctrl;
    int                i;

    ctrl = ucc_tl_shm_get_ctrl(seg, team, UCC_TL_TEAM_RANK(team));
    if (ctrl->ci != seq_num) {
        return UCC_INPROGRESS;
    }

    if (tree->top_tree) {
        for (i = 0; i < tree->top_tree->n_children; i++) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->top_tree->children[i]);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }

    if (tree->base_tree) {
        for (i = 0; i < tree->base_tree->n_children; i++) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->base_tree->children[i]);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_shm_reduce_seg_ready(ucc_tl_shm_seg_t *seg,
                                                       uint32_t seq_num,
                                                       ucc_tl_shm_team_t *team,
                                                       ucc_tl_shm_tree_t *tree)
{

    ucc_tl_shm_ctrl_t *ctrl;
    ucc_rank_t         parent;

    ctrl = ucc_tl_shm_get_ctrl(seg, team, UCC_TL_TEAM_RANK(team));
    if (ctrl->ci != seq_num) {
        return UCC_INPROGRESS;
    }

    if (tree->base_tree) {
        parent = tree->base_tree->parent;
        if (parent != UCC_RANK_INVALID) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }

    if (tree->top_tree) {
        parent = tree->top_tree->parent;
        if (parent != UCC_RANK_INVALID) {
            ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            if (ctrl->ci != seq_num) {
                return UCC_INPROGRESS;
            }
        }
    }
    return UCC_OK;
}

static inline void ucc_tl_shm_copy_to_children(ucc_tl_shm_seg_t * seg,
                                               ucc_tl_shm_team_t *team,
                                               ucc_kn_tree_t *    tree,
                                               uint32_t seq_num, int is_inline,
                                               void *src, size_t data_size)
{
    ucc_tl_shm_ctrl_t *ctrl;
    void *             dst;
    int                i;

    for (i = 0; i < tree->n_children; i++) {
        ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        dst  = is_inline ? ctrl->data
                         : ucc_tl_shm_get_data(seg, team, tree->children[i]);
        memcpy(dst, src, data_size);
        ucc_memory_cpu_store_fence();
        ctrl->pi = seq_num;
    }
}

static inline void ucc_tl_shm_signal_to_children(ucc_tl_shm_seg_t * seg,
                                                 ucc_tl_shm_team_t *team,
                                                 uint32_t           seq_num,
                                                 ucc_kn_tree_t *    tree)
{
    ucc_tl_shm_ctrl_t *ctrl;
    int                i;

    for (i = 0; i < tree->n_children; i++) {
        ctrl     = ucc_tl_shm_get_ctrl(seg, team, tree->children[i]);
        ctrl->pi = seq_num;
    }
}

#define UCC_TL_SHM_SET_SEG_READY_SEQ_NUM(_task, _team)                         \
    do {                                                                       \
        int _seg_id                = (_task)->seq_num % (_team)->n_concurrent; \
        (_task)->seg_ready_seq_num = (_team)->last_posted[_seg_id];            \
        (_team)->last_posted[_seg_id] = task->seq_num;                         \
    } while (0)
#endif
