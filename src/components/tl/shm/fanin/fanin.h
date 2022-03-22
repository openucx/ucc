/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef FANIN_H_
#define FANIN_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"

static inline ucc_status_t ucc_tl_shm_fanin_signal(ucc_tl_shm_team_t *team,
                                                   ucc_tl_shm_seg_t * seg,
                                                   ucc_tl_shm_task_t *task,
                                                   ucc_kn_tree_t *    tree)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl, *child_ctrl;
    ucc_rank_t         child;
    int                i, j;

    for (i = task->cur_child; i < tree->n_children; i++) {
        child      = tree->children[i];
        child_ctrl = ucc_tl_shm_get_ctrl(seg, team, child);
        for (j = 0; j < n_polls; j++) {
            if (child_ctrl->pi == seq_num) {
                break;
            }
        }
        if (j == n_polls) {
            task->cur_child = i;
            return UCC_INPROGRESS;
        }
    }
    if (tree->parent != UCC_RANK_INVALID) {
        /* signals to parent */
        my_ctrl     = ucc_tl_shm_get_ctrl(seg, team, team_rank);
        my_ctrl->pi = seq_num;
    }
    task->cur_child = 0;
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_fanin_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task_h);

#endif
