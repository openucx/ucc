/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "fanin.h"

ucc_status_t ucc_tl_shm_fanin_signal(ucc_tl_shm_team_t *team,
                                      ucc_tl_shm_seg_t *seg,
                                      ucc_tl_shm_task_t *task,
                                      ucc_kn_tree_t *tree)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl, *child_ctrl;
    ucc_rank_t         child;
    int                i, j;

    for (i = task->cur_child; i < tree->n_children; i++){
        child = tree->children[i];
        child_ctrl = ucc_tl_shm_get_ctrl(seg, team, child);
        for (j = 0; j < n_polls; j++) {
            if (child_ctrl->pi == seq_num) {
                SHMSEG_ISYNC();
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


ucc_status_t ucc_tl_shm_fanin_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl;

    if (!task->seg_ready) { //similar to reduce
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect fanin->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_reduce_seg_ready(seg, task->seg_ready_seq_num, team, tree)) {
            return UCC_INPROGRESS;
        }
        task->seg_ready = 1;
    }

    if (tree->base_tree && !task->first_tree_done) {
        status = ucc_tl_shm_fanin_signal(team, seg, task, tree->base_tree);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 1;
        task->cur_child = 0;
    }

    if (tree->top_tree) {
        status = ucc_tl_shm_fanin_signal(team, seg, task, tree->top_tree);

        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* fanin done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_fanin_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_fanin_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_fanin_start", 0);
    UCC_TL_SHM_SET_SEG_READY_SEQ_NUM(task, team);
    task->super.super.status = UCC_INPROGRESS;
    status = task->super.progress(&task->super);

    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_shm_fanin_init(ucc_tl_shm_task_t *task)
{
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
	ucc_rank_t   base_radix = UCC_TL_SHM_TEAM_LIB(team)->cfg.fanin_base_radix;
	ucc_rank_t   top_radix  = UCC_TL_SHM_TEAM_LIB(team)->cfg.fanin_top_radix;
	ucc_status_t status;

    task->super.post      = ucc_tl_shm_fanin_start;
    task->super.progress  = ucc_tl_shm_fanin_progress;
    task->seq_num         = team->seq_num++;
    task->seg             = &team->segs[task->seq_num % team->n_concurrent];
    task->seg_ready       = 0;
    task->cur_child       = 0;
    task->first_tree_done = 0;
    task->base_tree_only  = UCC_TL_SHM_TEAM_LIB(team)->cfg.base_tree_only;


    status = ucc_tl_shm_tree_init(team, args.root, base_radix, top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_FANIN,
                                  task->base_tree_only, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
    	return status;
    }
    return UCC_OK;
}
