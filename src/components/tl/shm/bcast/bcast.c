/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "bcast.h"

ucc_status_t ucc_tl_shm_bcast_write(ucc_tl_shm_team_t *team,
                                    ucc_tl_shm_seg_t *seg,
                                    ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    int *is_op_root, size_t data_size)
{
	ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl;
    void              *src;
    int                i;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->parent == UCC_RANK_INVALID) {
        /* i am root of the tree*/
        /* If the tree root is global OP root he can copy data out from
           origin user src buffer.
           Otherwise, it must be base_tree in 2lvl alg,
           and the data of the tree root is in the local shm (ctrl or data) */
        src = *is_op_root ? TASK_ARGS(task).src.info.buffer :
                           (is_inline ? my_ctrl->data :
                           ucc_tl_shm_get_data(seg, team, team_rank));
        ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline,
                                    src, data_size);
        return UCC_OK;
    }
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            SHMSEG_ISYNC();
            src = is_inline ? my_ctrl->data :
                              ucc_tl_shm_get_data(seg, team, team_rank);
            ucc_tl_shm_copy_to_children(seg, team, tree, seq_num,
                                        is_inline, src, data_size);
            /* copy out to user dest is done in the end of base bcast alg */
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_shm_bcast_read(ucc_tl_shm_team_t *team,
                                   ucc_tl_shm_seg_t *seg,
                                   ucc_tl_shm_task_t *task,
                                   ucc_kn_tree_t *tree, int is_inline,
                                   int *is_op_root, size_t data_size)
{
	ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
	uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void              *src, *dst;
    ucc_tl_shm_ctrl_t *parent_ctrl, *my_ctrl;
    ucc_rank_t         parent;
    int                i;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (*is_op_root) {
        /* Only global op root needs to copy the data from user src to its shm */
        if (*is_op_root == 1) {
            dst = is_inline ? my_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  team_rank);
            memcpy(dst, TASK_ARGS(task).src.info.buffer, data_size);
            SHMSEG_WMB();
            (*is_op_root)++;
        }
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }
    parent = tree->parent;
    if (parent == UCC_RANK_INVALID) {
        /* I'm the root of the tree and NOT is_op_root. It means the tree is
           base tree and i already have the data in my shm via top_tree step
           (read or write). Just notify children. */
//        ucc_assert(task->tree->base_tree == tree);
        ucc_assert(my_ctrl->pi == seq_num);
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }
    for (i = 0; i < n_polls; i++){
        if (my_ctrl->pi == seq_num) {
            parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            SHMSEG_ISYNC();
            if (tree == task->tree->top_tree || tree->n_children > 0) {
                src = is_inline ? parent_ctrl->data :
                                  ucc_tl_shm_get_data(seg, team, parent);
                dst = is_inline ? my_ctrl->data :
                                  ucc_tl_shm_get_data(seg, team,
                                                      team_rank);
                memcpy(dst, src, data_size);
                SHMSEG_WMB();
                ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
            }
            /* copy out to user dest is done in the end of base bcast alg */
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_shm_bcast_ww_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size = args.src.info.count *
                                   ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t) args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl;
    void              *src;

    if (!task->seg_ready && ((tree->base_tree && tree->base_tree->n_children > 0) || (tree->base_tree == NULL && tree->top_tree->n_children > 0))) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
    	status = ucc_tl_shm_bcast_seg_ready(seg, task->seq_num, team, tree);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->seg_ready = 1;
    }

    if (tree->top_tree && !task->first_tree_done) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree,
                                        is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 1;
    }

    if (tree->base_tree) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->base_tree,
                                        is_inline, &is_op_root, data_size);

        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step then the data is in the base_tree->parent SHM
       If we did WRITE as 2nd step then the data is in my SHM */
    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    if (!is_op_root) {
        src = is_inline ? my_ctrl->data : ucc_tl_shm_get_data(seg, team, rank);
        memcpy(args.src.info.buffer, src, data_size);
    }

    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_ww_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_wr_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size = args.src.info.count *
                                   ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t) args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         parent;
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl, *parent_ctrl;
    void              *src;

    if (!task->seg_ready && ((tree->base_tree && tree->base_tree->n_children > 0) || (tree->base_tree == NULL && tree->top_tree->n_children > 0))) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_bcast_seg_ready(seg, task->seq_num, team, tree)) {
            return UCC_INPROGRESS;
        }
        task->seg_ready = 1;
    }

//    volatile int flag = 0;
//    while (!flag) {}
    if (tree->top_tree && !task->first_tree_done) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree,
                                        is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 1;
    }

    if (tree->base_tree) {
        status = ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree, is_inline,
                                       &is_op_root, data_size);

        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    /* Copy out to user dest:
       - ranks that are non-roots in the base tree (those that have parent)
         will have the data in their parent's data/ctrl because we did READ
         on the last step.
       - other ranks must be participants of the top tree WRITE step, hence
        they have the data in their local shm data/ctrl */
    if (!is_op_root) {
        parent = tree->base_tree ? ((tree->base_tree->parent == UCC_RANK_INVALID) ?
                                            rank : tree->base_tree->parent) : rank;
        parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
        src = is_inline ? parent_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  parent);
        memcpy(args.src.info.buffer, src, data_size);
    }
    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_wr_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_rr_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size = args.src.info.count *
                                   ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t) args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t  *seg  = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline  = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    ucc_tl_shm_ctrl_t *my_ctrl, *parent_ctrl;
    ucc_rank_t         parent;
    ucc_status_t       status;
    void              *src;

    if (!task->seg_ready && ((tree->base_tree && tree->base_tree->n_children > 0) || (tree->base_tree == NULL && tree->top_tree->n_children > 0))) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_bcast_seg_ready(seg, task->seq_num, team, tree)) {
            return UCC_INPROGRESS;
        }
        task->seg_ready = 1;
    }

    if (tree->top_tree && !task->first_tree_done) {
        status = ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                                       is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 1;
    }

    if (tree->base_tree) {
        status = ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree,
                                      is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step then the data is in the base_tree->parent SHM
       If we did WRITE as 2nd step then the data is in my SHM */
    if (!is_op_root) {
        parent = (tree->base_tree && tree->base_tree->parent != UCC_RANK_INVALID) ?
                 tree->base_tree->parent : tree->top_tree->parent;
        parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
        src = is_inline ? parent_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  parent);
        memcpy(args.src.info.buffer, src, data_size);
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_rr_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_rw_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size = args.src.info.count *
                                   ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t) args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline  = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    ucc_tl_shm_ctrl_t *my_ctrl;
    ucc_status_t       status;
    void              *src;

    if (!task->seg_ready && ((tree->base_tree && tree->base_tree->n_children > 0) || (tree->base_tree == NULL && tree->top_tree->n_children > 0))) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_bcast_seg_ready(seg, task->seq_num, team, tree)) {
            return UCC_INPROGRESS;
        }
        task->seg_ready = 1;
    }

    if (tree->top_tree && !task->first_tree_done) {
        status = ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                                       is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
        task->first_tree_done = 1;
    }

    if (tree->base_tree) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->base_tree,
                                        is_inline, &is_op_root, data_size);

        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step then the data is in the base_tree->parent SHM
       If we did WRITE as 2nd step then the data is in my SHM */
    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    if (!is_op_root) {
        src = is_inline ? my_ctrl->data : ucc_tl_shm_get_data(seg, team, rank);
        memcpy(args.src.info.buffer, src, data_size);
    }

    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_rw_progress_done", 0);
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_start", 0);
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
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
	ucc_rank_t         base_radix = task->base_radix;
	ucc_rank_t         top_radix  = task->top_radix;
	ucc_status_t       status;

    task->super.post = ucc_tl_shm_bcast_start;
    task->seq_num    = team->seq_num++;
    task->seg        = &team->segs[task->seq_num % team->n_concurrent];
    task->first_tree_done = 0;
    task->seg_ready = 0;

    status = ucc_tl_shm_tree_init(team, args.root, base_radix, top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_BCAST,
                                  task->base_tree_only, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
    	return status;
    }

//    while(1) {}
//    switch(TASK_LIB(task)->cfg.bcast_alg) {
    switch(task->progress_alg) {
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
