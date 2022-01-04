/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "fanout.h"

ucc_status_t ucc_tl_shm_fanout_signal(ucc_tl_shm_team_t *team,
                                      ucc_tl_shm_seg_t *seg,
                                      ucc_tl_shm_task_t *task,
                                      ucc_kn_tree_t *tree,
                                      int is_op_root)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl;
    int                i;

    if (is_op_root) {
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            SHMSEG_ISYNC();
            ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}


ucc_status_t ucc_tl_shm_fanout_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_rank_t         root = (ucc_rank_t) args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_op_root = rank == root;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl;

    if (is_op_root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect fanout->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) { //TODO: implement
            return UCC_INPROGRESS;
        }
    }

    if (tree->top_tree) {
        status = ucc_tl_shm_fanout_signal(team, seg, task, tree->top_tree,
                                          is_op_root);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }
    status = ucc_tl_shm_fanout_signal(team, seg, task, tree->base_tree,
                                      is_op_root);

    if (UCC_OK != status) {
        /* in progress */
        return status;
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* fanout done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_fanout_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_fanout_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_fanout_start", 0);
    task->seq_num++;
    task->super.super.status = UCC_INPROGRESS;
    status = task->super.progress(&task->super);

    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_shm_fanout_init(ucc_tl_shm_task_t *task)
{
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
	ucc_rank_t   base_radix = UCC_TL_SHM_TEAM_LIB(team)->cfg.fanout_base_radix;
//	ucc_rank_t   base_radix = task->base_radix;
	ucc_rank_t   top_radix  = UCC_TL_SHM_TEAM_LIB(team)->cfg.fanout_top_radix;
	ucc_status_t status;

    task->super.post = ucc_tl_shm_fanout_start;
    task->super.progress = ucc_tl_shm_fanout_progress;
    task->seq_num    = team->seq_num++;
    task->seg        = &team->segs[task->seq_num % team->n_concurrent];
    status = ucc_tl_shm_tree_init(team, args.root, base_radix, top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_FANOUT,
                                  &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
    	return status;
    }
    return UCC_OK;
}
