/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_shm.h"
#include "reduce.h"

//ucc_status_t ucc_tl_shm_reduce_write(ucc_tl_shm_team_t *team,
//                                    ucc_tl_shm_seg_t *seg,
//                                    ucc_tl_shm_task_t *task,
//                                    ucc_kn_tree_t *tree, int is_inline,
//                                    int *is_op_root, size_t data_size,
//                                    ucc_memory_type_t mtype)
//{
//	ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
//    uint32_t           seq_num   = task->seq_num;
//    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
//    ucc_tl_shm_ctrl_t *my_ctrl;
//    void              *src;
//    int                i;
//
//    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);
//
//    if (tree->parent == UCC_RANK_INVALID) {
//        /* i am root of the tree*/
//        /* If the tree root is global OP root he can copy data out from
//           origin user src buffer.
//           Otherwise, it must be base_tree in 2lvl alg,
//           and the data of the tree root is in the local shm (ctrl or data) */
//        src = *is_op_root ? TASK_ARGS(task).src.info.buffer :
//                           (is_inline ? my_ctrl->data :
//                           ucc_tl_shm_get_data(seg, team, team_rank));
//        ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline,
//                                    src, data_size);
//        return UCC_OK;
//    }
//    for (i = 0; i < n_polls; i++) {
//        if (my_ctrl->pi == seq_num) {
//            SHMSEG_ISYNC();
//            src = is_inline ? my_ctrl->data :
//                              ucc_tl_shm_get_data(seg, team, team_rank);
//            ucc_tl_shm_copy_to_children(seg, team, tree, seq_num,
//                                        is_inline, src, data_size);
//            /* copy out to user dest is done in the end of base bcast alg */
//            return UCC_OK;
//        }
//    }
//    return UCC_INPROGRESS;
//}

ucc_status_t ucc_tl_shm_reduce_read(ucc_tl_shm_team_t *team,
                                    ucc_tl_shm_seg_t *seg,
                                    ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    size_t count, ucc_datatype_t dt,
                                    ucc_memory_type_t mtype,
                                    ucc_coll_args_t *args)
{
	ucc_rank_t         team_rank  = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls    = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void              *src1, *src2, *dst;
    ucc_tl_shm_ctrl_t *child_ctrl, *my_ctrl;
    ucc_rank_t         child;
    int                i, j, reduced;
    ucc_status_t       status;


    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->n_children == 0) {
        /* I am leaf so I dont need to read, only notify parent*/

        if (tree == task->tree->base_tree || task->tree->base_tree == NULL) {
        /* I am leaf in base tree so need to copy from user buffer into my shm */
            dst = is_inline ? my_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  team_rank);
            memcpy(dst, args->src.info.buffer, count * ucc_dt_size(dt));
            SHMSEG_WMB();
        }
        my_ctrl->pi = seq_num; //signals to parent
        return UCC_OK;
    }

    for (i = task->cur_child; i < tree->n_children; i++) {
    	reduced = 0;
        child = tree->children[i];
        child_ctrl = ucc_tl_shm_get_ctrl(seg, team, child);
        for (j = 0; j < n_polls; j++) {
            if (child_ctrl->pi == seq_num) {
                SHMSEG_ISYNC();
                src1 = is_inline ? child_ctrl->data :
                                   ucc_tl_shm_get_data(seg, team, child);
                dst  = (args->root == team_rank) ? args->dst.info.buffer : (is_inline ? my_ctrl->data :
                                   ucc_tl_shm_get_data(seg, team, team_rank));
                src2 = (task->first_reduce) ? args->src.info.buffer : dst;
                status = ucc_dt_reduce(src1, src2, dst, count, dt, mtype, args);

                if (ucc_unlikely(UCC_OK != status)) {
                    tl_error(UCC_TASK_LIB(task),
                             "failed to perform dt reduction");
                    task->super.super.status = status;
                    return status;
                }
                SHMSEG_WMB();
                task->first_reduce = 0;
                reduced = 1;
                break;
            }
        }
        if (!reduced) {
        	task->cur_child = i;
            return UCC_INPROGRESS;
        }
    }
    my_ctrl->pi = seq_num; //signals to parent
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_reduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);

    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    ucc_rank_t         root = (ucc_rank_t) args.root;
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline;
    int                is_op_root = rank == root;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl;

    if (is_op_root) {
        count = args.dst.info.count;
        mtype = args.dst.info.mem_type;
        dt    = args.dst.info.datatype;
    } else {
        count = args.src.info.count;
        mtype = args.src.info.mem_type;
        dt    = args.src.info.datatype;
    }
    data_size = count * ucc_dt_size(dt);
    is_inline = data_size <= team->max_inline;

    if (!task->seg_ready) {
        /* checks if previous collective has completed on the seg
        TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_reduce_seg_ready(seg, task->seq_num, team, tree)) {
            return UCC_INPROGRESS;
        }
        task->seg_ready = 1;
    }

    if (tree->base_tree && !task->first_tree_done) {
        status = ucc_tl_shm_reduce_read(team, seg, task, tree->base_tree,
                                    is_inline, count, dt, mtype, &args);

        if (UCC_OK != status) {
            /* in progress or reduction failed */
            return status;
        }
        task->first_tree_done = 1;
        task->cur_child = 0;
    }

    if (tree->top_tree) {
        status = ucc_tl_shm_reduce_read(team, seg, task, tree->top_tree,
                                        is_inline, count, dt, mtype, &args);
        if (UCC_OK != status) {
            /* in progress or reduction failed */
            return status;
        }
    }

    /* Copy out to user dest:
       - ranks that are non-roots in the base tree (those that have parent)
         will have the data in their parent's data/ctrl because we did READ
         on the last step.
       - other ranks must be participants of the top tree WRITE step, hence
        they have the data in their local shm data/ctrl */

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;

    /* reduce done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_reduce_rr_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_reduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_status_t       status;

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_reduce_start", 0);
    task->super.super.status = UCC_INPROGRESS;
    status = task->super.progress(&task->super);

    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_shm_reduce_init(ucc_tl_shm_task_t *task)
{
 	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
	ucc_rank_t         base_radix = task->base_radix;
	ucc_rank_t         top_radix  = task->top_radix;
	ucc_status_t       status;

    if (args.op == UCC_OP_AVG) {
        task->super.super.status = UCC_ERR_NOT_SUPPORTED;
    	return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_shm_reduce_start;
    task->super.progress = ucc_tl_shm_reduce_progress;


    task->seq_num    = team->seq_num++;
    task->seg        = &team->segs[task->seq_num % team->n_concurrent];
//    task->top_cur_child  = 0;
//    task->base_cur_child = 0;
    task->cur_child = 0;
    task->first_reduce   = 1;
    task->first_tree_done  = 0;
    task->seg_ready  = 0;

//    task->base_tree_done = 0;
//    task->progress_in_top_tree = 0;
    status = ucc_tl_shm_tree_init(team, args.root, base_radix, top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_REDUCE,
                                  task->base_tree_only, &task->tree);

//    while(1) {}
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
    	return status;
    }

//    switch(TASK_LIB(task)->cfg.reduce_alg) {
//        case REDUCE_WW:
//        	task->super.progress = ucc_tl_shm_reduce_ww_progress;
//            break;
//        case REDUCE_WR:
//        	task->super.progress = ucc_tl_shm_reduce_wr_progress;
//            break;
//        case REDUCE_RR:
//        	task->super.progress = ucc_tl_shm_reduce_rr_progress;
//            break;
//        case REDUCE_RW:
//        	task->super.progress = ucc_tl_shm_reduce_rw_progress;
//            break;
//    }
    return UCC_OK;
}
