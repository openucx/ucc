/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "bcast.h"
#include "utils/arch/cpu.h"

enum
{
    BCAST_STAGE_START,
    BCAST_STAGE_BASE_TREE,
    BCAST_STAGE_TOP_TREE,
    BCAST_STAGE_CI,
};

ucc_status_t ucc_tl_shm_bcast_write(ucc_tl_shm_team_t *team,
                                    ucc_tl_shm_seg_t * seg,
                                    ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    int *is_op_root, size_t data_size)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl;
    void *             src;
    int                i;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->parent == UCC_RANK_INVALID) {
        /* i am root of the tree*/
        /* If the tree root is global OP root he can copy data out from
           origin user src buffer.
           Otherwise, it must be base_tree in 2lvl alg,
           and the data of the tree root is in the local shm (ctrl or data) */
        src = *is_op_root
                  ? TASK_ARGS(task).src.info.buffer
                  : (is_inline ? my_ctrl->data
                               : ucc_tl_shm_get_data(seg, team, team_rank));
        ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline, src,
                                    data_size);
        return UCC_OK;
    }
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            src = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, team_rank);
            ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline,
                                        src, data_size);
            /* copy out to user dest is done in the end of base bcast alg */
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_shm_bcast_read(ucc_tl_shm_team_t *team,
                                   ucc_tl_shm_seg_t * seg,
                                   ucc_tl_shm_task_t *task,
                                   ucc_kn_tree_t *tree, int is_inline,
                                   int *is_op_root, size_t data_size)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void *             src, *dst;
    ucc_tl_shm_ctrl_t *parent_ctrl, *my_ctrl;
    ucc_rank_t         parent;
    int                i;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (*is_op_root) {
        /* Only global op root needs to copy the data from user src to its shm */
        if (*is_op_root == 1) {
            dst = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, team_rank);
            memcpy(dst, TASK_ARGS(task).src.info.buffer, data_size);
            ucc_memory_cpu_store_fence();
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
        ucc_assert(my_ctrl->pi == seq_num);
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
        return UCC_OK;
    }
    for (i = 0; i < n_polls; i++) {
        if (my_ctrl->pi == seq_num) {
            parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            if (tree == task->tree->top_tree || tree->n_children > 0) {
                src = is_inline ? parent_ctrl->data
                                : ucc_tl_shm_get_data(seg, team, parent);
                dst = is_inline ? my_ctrl->data
                                : ucc_tl_shm_get_data(seg, team, team_rank);
                memcpy(dst, src, data_size);
                ucc_memory_cpu_store_fence();
                ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
            }
            /* copy out to user dest is done in the end of base bcast alg */
            return UCC_OK;
        }
    }
    return UCC_INPROGRESS;
}

static void ucc_tl_shm_bcast_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size =
        args.src.info.count * ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t)args.root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         parent;
    ucc_tl_shm_seg_t * seg        = task->seg;
    ucc_tl_shm_tree_t *tree       = task->tree;
    int                is_inline  = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    ucc_tl_shm_ctrl_t *my_ctrl, *parent_ctrl;
    void *             src;
    int                i;

next_stage:
    switch (task->stage) {
    case BCAST_STAGE_START:
        if ((tree->base_tree && tree->base_tree->n_children > 0) ||
            (tree->base_tree == NULL && tree->top_tree->n_children > 0)) {
            /* checks if previous collective has completed on the seg
                TODO: can be optimized if we detect bcast->reduce pattern.*/
            SHMCHECK_GOTO(ucc_tl_shm_bcast_seg_ready(seg,
                          task->seg_ready_seq_num, team, tree), task, out);
        }
        if (tree->top_tree) {
            task->stage = BCAST_STAGE_TOP_TREE;
        } else {
            task->stage = BCAST_STAGE_BASE_TREE;
        }
        goto next_stage;
    case BCAST_STAGE_TOP_TREE:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_WR) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree,
                          is_inline, &is_op_root, data_size), task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        if (tree->base_tree) {
            task->stage = BCAST_STAGE_BASE_TREE;
            goto next_stage;
        }
        break;
    case BCAST_STAGE_BASE_TREE:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_RW) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task, tree->base_tree,
                          is_inline, &is_op_root, data_size), task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        break;
    case BCAST_STAGE_CI:
        goto ci;
        break;
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step then the data is in the base_tree->parent SHM
       If we did WRITE as 2nd step then the data is in my SHM */

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);

    if (!is_op_root) {
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_RW) {
            src = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, rank);
        } else {
            if (task->progress_alg == BCAST_RR) {
                parent = (tree->base_tree &&
                          tree->base_tree->parent != UCC_RANK_INVALID)
                             ? tree->base_tree->parent
                             : tree->top_tree->parent;
            } else {
                parent = tree->base_tree
                             ? ((tree->base_tree->parent == UCC_RANK_INVALID)
                                    ? rank
                                    : tree->base_tree->parent)
                             : rank;
            }
            parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
            src         = is_inline ? parent_ctrl->data
                                    : ucc_tl_shm_get_data(seg, team, parent);
        }
        memcpy(args.src.info.buffer, src, data_size);
        ucc_memory_cpu_store_fence();
    }

    if (!is_op_root && tree->top_tree && tree->base_tree &&
        tree->base_tree->parent == UCC_RANK_INVALID &&
        task->progress_alg == BCAST_WR) {
        /* This handles a special case of potential race:
           it only can happen when algorithm is WR and 2 trees are used.
           Socket leader which is not actual op root must wait for its
           children to complete reading the data from its SHM before it
           can set its own CI (signalling the seg can be re-used).

           Otherwise, parent rank of this socket leader in the top tree
           (either actual root or another socket leader) may overwrite the
           SHM data in the subsequent bcast, while the data is not entirely
           copied by leafs.
        */
        task->stage = BCAST_STAGE_CI;
    ci:
        for (i = 0; i < tree->base_tree->n_children; i++) {
            ucc_tl_shm_ctrl_t *ctrl =
                ucc_tl_shm_get_ctrl(seg, team, tree->base_tree->children[i]);
            if (ctrl->ci < task->seq_num) {
                return;
            }
        }
        my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    }
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_rw_progress_done",
                                     0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_bcast_start", 0);
    UCC_TL_SHM_SET_SEG_READY_SEQ_NUM(task, team);
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_shm_bcast_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     tl_team,
                                   ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_tl_shm_task_t *task;
    ucc_status_t       status;

    if (UCC_IS_PERSISTENT(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_shm_get_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    team->perf_params_bcast(&task->super);

    task->super.post     = ucc_tl_shm_bcast_start;
    task->super.progress = ucc_tl_shm_bcast_progress;
    task->stage          = BCAST_STAGE_START;

    status = ucc_tl_shm_tree_init(team, coll_args->args.root, task->base_radix,
                                  task->top_radix, &task->tree_in_cache,
                                  UCC_COLL_TYPE_BCAST, task->base_tree_only,
                                  &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
        return status;
    }

    *task_h = &task->super;
    return UCC_OK;
}
