/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_shm.h"
#include "bcast.h"

ucc_status_t ucc_tl_shm_bcast_wr_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t data_size = args.src.info.count *
                       ucc_dt_size(args.src.info.datatype);
//    int64_t sequence_number = shm_seg_data->seq_num;
    ucc_rank_t root = (ucc_rank_t) args.root, group_size = UCC_TL_TEAM_SIZE(team), rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int is_inline = data_size <= max_inline;
    int leaders_group_mode = team->cfg.leaders_group_mode; //what os this mode?
    int base_group_mode = team->cfg.base_group_mode;
    if (rank == root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) {
            return UCC_INPROGRESS:
        }
    }
    if (tree->top_tree) {
        status = do_bcast_write(seg, task, top_tree, is_inline, rank == root,
                                data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }
    status = do_bcast_read(seg, task, base_tree, is_inline, rank == root,
                           data_size);
    if (UCC_OK != status) {
        /* in progress */
        return status;
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step then the data is in the base_tree->parent SHM
       If we did WRITE as 2nd step then the data is in my SHM */
    if (rank != root) {
        parent_ctrl = get_ctrl(seg, team, tree->parent);
        void *src = is_inline ? parent_ctrl->data : get_data(seg, team, tree->parent));
        memcpy(args.src.info.buffer, src, data_size);
    }

    my_ctrl->ci = seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
    void *ptr;
    ucc_status_t status;
    size_t data_size = args.src.info.count *
                       ucc_dt_size(args.src.info.datatype);
    ucc_rank_t rank = UCC_TL_TEAM_RANK(team);
    //int on_node_rank = team->shm_seg.on_node_rank;

    task->seq_num++;

//    task->shmseg_dest = (void *)ptr;
//    if (args.root == rank) {
//    	memcpy(ptr, args.src.info.buffer, data_size); //something instead?
//    }
    task->super.super.status = UCC_INPROGRESS;

    status = task->super.progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}


ucc_status_t ucc_tl_shm_bcast_init(ucc_tl_shm_task_t *task)
{
	ucc_tl_shm_team_t *team  = TASK_TEAM(task);
	ucc_coll_args_t    args  = TASK_ARGS(task);
	ucc_rank_t         radix = UCC_TL_SHM_TEAM_LIB(team)->cfg.bcast_kn_radix;
    task->super.post = ucc_tl_shm_bcast_start;
    task->seq_num    = team->seq_num++;
    task->seg        = team->segs[task->seq_num % team->n_concurrent];
    ucc_tl_shm_tree_init_bcast(team, args.root, radix, &task->tree); //should happen in init ot start?
//    task->super.progress = ucc_tl_shm_bcast_progress;
    switch(TASK_LIB(task)->cfg.bcast_alg) {
        case BCAST_WW:
        	task->super.progress = ucc_tl_shm_bcast_ww_progress;
            break;
        case BCAST_WR:
        	task->super.progress = ucc_tl_shm_bcast_wr_progress;
            break;
        case BCAST_RR:
        	task->super.progress = ucc_tl_shm_bcast_rr_progress;
            break;
        case BCAST_RW:
        	task->super.progress = ucc_tl_shm_bcast_rw_progress;
            break;
    }
    return UCC_OK;
}
