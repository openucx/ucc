/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "allreduce.h"

enum
{
    ALLREDUCE_STAGE_START,
    ALLREDUCE_STAGE_BASE_TREE_REDUCE,
    ALLREDUCE_STAGE_TOP_TREE_REDUCE,
    ALLREDUCE_STAGE_BASE_TREE_BCAST,
    ALLREDUCE_STAGE_TOP_TREE_BCAST,
    ALLREDUCE_BCAST_STAGE_CI,
};

static void ucc_tl_shm_allreduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task       = ucc_derived_of(coll_task,
                                                   ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team       = TASK_TEAM(task);
    ucc_coll_args_t    args       = TASK_ARGS(task);
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         root       = ucc_ep_map_eval(team->base_groups[ucc_ep_map_eval(team->rank_group_id_map, UCC_TL_TEAM_RANK(team))].map, 0);
    ucc_tl_shm_seg_t  *seg        = task->seg;
    ucc_tl_shm_tree_t *tree       = task->tree;
    ucc_memory_type_t  mtype      = args.dst.info.mem_type;
    ucc_datatype_t     dt         = args.dst.info.datatype;
    size_t             count      = args.dst.info.count;
    size_t             data_size  = count * ucc_dt_size(dt);
    int                is_inline  = data_size <= team->max_inline;
    int                is_op_root = rank == root;
    int                i;
    ucc_rank_t         parent;
    ucc_tl_shm_ctrl_t *my_ctrl, *parent_ctrl;
    void              *src;

next_stage:
    switch (task->stage) {
    case ALLREDUCE_STAGE_START:
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        SHMCHECK_GOTO(ucc_tl_shm_reduce_seg_ready(seg, task->seg_ready_seq_num,
                                                  team, tree), task, out);
        if (tree->base_tree) {
            task->stage = ALLREDUCE_STAGE_BASE_TREE_REDUCE;
        } else {
            task->stage = ALLREDUCE_STAGE_TOP_TREE_REDUCE;
        }
        goto next_stage;
    case ALLREDUCE_STAGE_BASE_TREE_REDUCE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, tree->base_tree,
                      is_inline, count, dt, mtype, &args), task, out);
        task->cur_child = 0;
        if (tree->top_tree) {
            task->stage = ALLREDUCE_STAGE_TOP_TREE_REDUCE;
        } else {
            if (is_op_root) {
                memcpy(args.src.info.buffer, args.dst.info.buffer, data_size); //need this memcpy because root in bcast memcpys directly from src buffer. Is this allowed?
                ucc_memory_cpu_store_fence();
            }
            task->stage = ALLREDUCE_STAGE_BASE_TREE_BCAST;
            task->seq_num++; /* finished reduce, need seq_num to be updated for bcast */
        }
        goto next_stage;
    case ALLREDUCE_STAGE_TOP_TREE_REDUCE:
        SHMCHECK_GOTO(ucc_tl_shm_reduce_read(team, seg, task, tree->top_tree,
                      is_inline, count, dt, mtype, &args), task, out);
        if (is_op_root) {
            memcpy(args.dst.info.buffer, args.src.info.buffer, data_size);
            ucc_memory_cpu_store_fence();
        }
        task->stage = ALLREDUCE_STAGE_TOP_TREE_BCAST;
        task->seq_num++; /* finished reduce, need seq_num to be updated for bcast */
        goto next_stage;
    case ALLREDUCE_STAGE_TOP_TREE_BCAST:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_WR) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree,
                          is_inline, &is_op_root, data_size), task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        if (tree->base_tree) {
            task->stage = ALLREDUCE_STAGE_BASE_TREE_BCAST;
            goto next_stage;
        }
        break;
    case ALLREDUCE_STAGE_BASE_TREE_BCAST:
        if (task->progress_alg == BCAST_WW || task->progress_alg == BCAST_RW) {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_write(team, seg, task, tree->base_tree,
                          is_inline, &is_op_root, data_size), task, out);
        } else {
            SHMCHECK_GOTO(ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree,
                          is_inline, &is_op_root, data_size), task, out);
        }
        break;
    case ALLREDUCE_BCAST_STAGE_CI:
        goto ci;
        break;
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step in bcast then the data is in the
       base_tree->parent SHM.
       If we did WRITE as 2nd step in bcast then the data is in my SHM */

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
        memcpy(args.dst.info.buffer, src, data_size); // changed memcpy into args.dst.info.buffer (instead of src buffer) to fit allreduce api and save additional memcpy
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
        task->stage = ALLREDUCE_BCAST_STAGE_CI;
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

//    if (!is_op_root) {
//        memcpy(args.dst.info.buffer, args.src.info.buffer, data_size);
//        ucc_memory_cpu_store_fence();
//    }

    /* task->seq_num was updated between reduce and bcast, now needs to be
       rewinded to fit general collectives order, as allreduce is actually
        a single collective */
    my_ctrl->ci = task->seq_num - 1;
    /* allreduce done */
    task->super.status = UCC_OK;
    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task,
                                     "shm_allreduce_progress_done", 0);
out:
    return;
}

static ucc_status_t ucc_tl_shm_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);

    UCC_TL_SHM_PROFILE_REQUEST_EVENT(coll_task, "shm_allreduce_start", 0);
    UCC_TL_SHM_SET_SEG_READY_SEQ_NUM(task, team);
    task->super.status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_shm_allreduce_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     tl_team,
                                     ucc_coll_task_t **    task_h)
{
    ucc_tl_shm_team_t *team = ucc_derived_of(tl_team, ucc_tl_shm_team_t);
    ucc_rank_t base_radix   = UCC_TL_SHM_TEAM_LIB(team)->cfg.allreduce_base_radix;
    ucc_rank_t top_radix    = UCC_TL_SHM_TEAM_LIB(team)->cfg.allreduce_top_radix;
    ucc_rank_t root         = 0;
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
    task->root = ucc_ep_map_eval(team->base_groups[ucc_ep_map_eval(team->rank_group_id_map, UCC_TL_TEAM_RANK(team))].map, 0);

    /*TODO: make sure both bcast and reduce perf params are applicable - currently cant because each set base/top radix and base_tree_only,
            which will be used for tree init. Only flag that can be changed at critical section is bcast progress alg */
//    team->perf_params_bcast(&task->super);
//    team->perf_params_reduce(&task->super);

    task->super.post     = ucc_tl_shm_allreduce_start;
    task->super.progress = ucc_tl_shm_allreduce_progress;
    task->stage          = ALLREDUCE_STAGE_START;

    status = ucc_tl_shm_tree_init(team, root, base_radix, top_radix,
                                  &task->tree_in_cache, UCC_COLL_TYPE_REDUCE,
                                  task->base_tree_only, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
