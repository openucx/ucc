/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "barrier.h"

ucc_status_t ucc_tl_shm_barrier_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t  *seg  = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl;

    if (!task->seg_ready) { //similar to reduce
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect barrier->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_reduce_seg_ready(seg, task->seq_num, team, tree)) {
            return UCC_INPROGRESS;
        }
        task->seg_ready = 1;
    }

    if (!task->first_tree_done) {
        if (tree->base_tree) {
            status = ucc_tl_shm_fanin_signal(team, seg, task, tree->base_tree);
            if (UCC_OK != status) {
                /* in progress */
                return status;
            }
        }
        task->first_tree_done = 1;
        task->cur_child = 0;
    }

    if (tree->top_tree &&  task->first_tree_done == 1) {
        status = ucc_tl_shm_fanin_signal(team, seg, task, tree->top_tree);

        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 2;
        task->cur_child = 0;
    }

    if (!task->barrier_fanin_done){
        task->seq_num++;
        task->barrier_fanin_done = 1;
	}

    if (tree->top_tree && task->first_tree_done == 2) {
        status = ucc_tl_shm_fanout_signal(team, seg, task, tree->top_tree);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 3;
        task->cur_child = 0;
    }

    if (tree->base_tree) {
        status = ucc_tl_shm_fanout_signal(team, seg, task, tree->base_tree);

        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num - 1;
    /* barrier done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_barrier_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_barrier_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_barrier_start", 0);
    task->super.super.status = UCC_INPROGRESS;
    status = task->super.progress(&task->super);

//    task->barrier_fanin_done = 0;

    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_shm_barrier_init(ucc_tl_shm_task_t *task)
{
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_rank_t   base_radix = UCC_TL_SHM_TEAM_LIB(team)->cfg.barrier_base_radix;
	ucc_rank_t   top_radix  = UCC_TL_SHM_TEAM_LIB(team)->cfg.barrier_top_radix;
	ucc_rank_t   root = 0;
	ucc_status_t status;

    task->super.post      = ucc_tl_shm_barrier_start;
    task->super.progress  = ucc_tl_shm_barrier_progress;
    task->seq_num         = team->seq_num++;
    task->seg             = &team->segs[task->seq_num % team->n_concurrent];
    task->seg_ready       = 0;
    task->cur_child       = 0;
    task->first_tree_done = 0;
    task->barrier_fanin_done = 0;
    task->base_tree_only  = UCC_TL_SHM_TEAM_LIB(team)->cfg.base_tree_only;

    status = ucc_tl_shm_tree_init(team, root, base_radix, top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_REDUCE,
                                  task->base_tree_only, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
    	return status;
    }
    return UCC_OK;
}
