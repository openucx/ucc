/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_shm.h"
#include "reduce.h"

enum
{
    REDUCE_STAGE_START,
    REDUCE_STAGE_BASE_TREE,
    REDUCE_STAGE_TOP_TREE,
};

ucc_status_t
ucc_tl_shm_reduce_read(ucc_tl_shm_team_t *team, ucc_tl_shm_seg_t *seg,
                       ucc_tl_shm_task_t *task, ucc_kn_tree_t *tree,
                       int is_inline, size_t count, ucc_datatype_t dt,
                       ucc_memory_type_t mtype, ucc_coll_args_t *args)
{
    ucc_rank_t         team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t           seq_num   = task->seq_num;
    uint32_t           n_polls   = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void *             src1, *src2, *dst;
    ucc_tl_shm_ctrl_t *child_ctrl, *my_ctrl;
    ucc_rank_t         child;
    int                i, j, reduced;
    ucc_status_t       status;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->n_children == 0) {
        /* I am leaf so I dont need to read, only notify parent*/

        if (tree == task->tree->base_tree || task->tree->base_tree == NULL) {
            /* I am leaf in base tree so need to copy from user buffer into my shm */
            dst = is_inline ? my_ctrl->data
                            : ucc_tl_shm_get_data(seg, team, team_rank);
            memcpy(dst, UCC_IS_INPLACE(*args) ? args->dst.info.buffer
                        : args->src.info.buffer, count * ucc_dt_size(dt));
            ucc_memory_cpu_store_fence();
        }
        my_ctrl->pi = seq_num; //signals to parent
        return UCC_OK;
    }

    for (i = task->cur_child; i < tree->n_children; i++) {
        reduced    = 0;
        child      = tree->children[i];
        child_ctrl = ucc_tl_shm_get_ctrl(seg, team, child);
        for (j = 0; j < n_polls; j++) {
            if (child_ctrl->pi == seq_num) {
                src1   = is_inline ? child_ctrl->data
                                   : ucc_tl_shm_get_data(seg, team, child);
                dst    = (task->root == team_rank)
                             ? args->dst.info.buffer
                             : (is_inline
                                    ? my_ctrl->data
                                    : ucc_tl_shm_get_data(seg, team, team_rank));
                src2   = (task->first_reduce)
                             ? ((UCC_IS_INPLACE(*args) &&
                                 task->root == team_rank) ?
                                 args->dst.info.buffer : args->src.info.buffer)
                             : dst;
                status = ucc_dt_reduce(src1, src2, dst, count, dt, mtype, args);

                if (ucc_unlikely(UCC_OK != status)) {
                    tl_error(UCC_TASK_LIB(task),
                             "failed to perform dt reduction");
                    task->super.super.status = status;
                    return status;
                }
                ucc_memory_cpu_store_fence();
                task->first_reduce = 0;
                reduced            = 1;
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

static void ucc_tl_shm_reduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);

    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    ucc_rank_t         root = task->root;
    ucc_tl_shm_seg_t * seg  = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline;
    int                is_op_root = rank == root;
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

next_stage:
    switch (task->stage) {
    case REDUCE_STAGE_START:
        /* checks if previous collective has completed on the seg
        TODO: can be optimized if we detect bcast->reduce pattern.*/
        SHMCHECK_GOTO(ucc_tl_shm_reduce_seg_ready(seg, task->seg_ready_seq_num,
                                                  team, tree), task, out);
        if (tree->base_tree) {
            task->stage = REDUCE_STAGE_BASE_TREE;
        } else {
            task->stage = REDUCE_STAGE_TOP_TREE;
        }
        goto next_stage;
    case REDUCE_STAGE_BASE_TREE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, tree->base_tree,
                      is_inline, count, dt, mtype, &args), task, out);
        task->cur_child = 0;
        if (tree->top_tree) {
            task->stage = REDUCE_STAGE_TOP_TREE;
            goto next_stage;
        }
        break;
    case REDUCE_STAGE_TOP_TREE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, tree->top_tree,
                      is_inline, count, dt, mtype, &args), task, out);
        break;
    }

    my_ctrl     = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;

    /* reduce done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_reduce_rr_done", 0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_reduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_reduce_start", 0);
    UCC_TL_SHM_SET_SEG_READY_SEQ_NUM(task, team);
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_shm_reduce_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *     tl_team,
                                    ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_tl_shm_task_t *task;
    ucc_status_t       status;

    if (UCC_IS_PERSISTENT(coll_args->args) ||
        coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_shm_get_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    team->perf_params_reduce(&task->super);

    task->super.post     = ucc_tl_shm_reduce_start;
    task->super.progress = ucc_tl_shm_reduce_progress;
    task->stage          = REDUCE_STAGE_START;

    status = ucc_tl_shm_tree_init(team, task->root, task->base_radix,
                                  task->top_radix, &task->tree_in_cache,
                                  UCC_COLL_TYPE_REDUCE, task->base_tree_only,
                                  &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
