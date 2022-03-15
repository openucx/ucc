/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef FANOUT_H_
#define FANOUT_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"


static inline ucc_status_t
ucc_tl_shm_fanout_signal(ucc_tl_shm_team_t *team,
                         ucc_tl_shm_seg_t *seg,
                         ucc_tl_shm_task_t *task,
                         ucc_kn_tree_t *tree)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl;
    int                i;

    if (tree->parent == UCC_RANK_INVALID) {
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_shm_fanout_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t      *team,
                                    ucc_coll_task_t     **task_h);

#endif
