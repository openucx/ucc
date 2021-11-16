/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "../tl_shm.h"
#include "bcast.h"

ucc_status_t ucc_tl_shm_tree_init_bcast(ucc_tl_shm_team_t *team,
                                        ucc_rank_t root,
                                        ucc_rank_t base_radix,
                                        ucc_rank_t top_radix,
                                        ucc_tl_shm_tree_t **tree_p)
{
//    ucc_kn_tree_t *base_tree, *top_tree;
    ucc_tl_shm_tree_t *shm_tree;
    ucc_rank_t tree_root, rank;
    size_t top_tree_size, base_tree_size;
    ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t group_size = team->base_groups[team->my_group_id].group_size;
    ucc_rank_t leaders_size = team->leaders_group->group_size;

    if (ucc_tl_shm_cache_tree_lookup(team, base_radix, top_radix, root, UCC_COLL_TYPE_BCAST, tree_p) == 1) {
        return UCC_OK;
    }
    /* Pool is initialized using UCC_KN_TREE_SIZE macro memory estimation, using
       base_group[my_group_id]->size and max supported radix (maybe up to group size as well */
    top_tree_size = ucc_tl_shm_kn_tree_size(leaders_size, top_radix);
    base_tree_size = ucc_tl_shm_kn_tree_size(group_size, base_radix);
    shm_tree = (ucc_tl_shm_tree_t *) ucc_malloc(sizeof(ucc_kn_tree_t) *
                                     (top_tree_size + base_tree_size));//TODO: alloc_from_shm_tree_pool() ucc_mpool_get(&ctx->req_mp);

    if (!shm_tree) {
        return UCC_ERR_NO_MEMORY;
    }

    shm_tree->top_tree = PTR_OFFSET(shm_tree, sizeof(ucc_kn_tree_t) * base_tree_size);

    ucc_rank_t root_group = ucc_ep_map_eval(team->rank_group_id_map, root);
    if (group_size > 1) {
//        size = team->base_groups[team->my_group_id]->group_size;
        rank = team->base_groups[team->my_group_id].group_rank;
        if (team->my_group_id == root_group) {
            tree_root = ucc_ep_map_eval(team->group_rank_map, root);
        } else {
            tree_root = 0;
        }
        ucc_tl_shm_kn_tree_init(group_size, tree_root, rank, base_radix,
                                UCC_COLL_TYPE_BCAST, shm_tree->base_tree);
//        shm_tree->base_tree = base_tree;
        /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
        ucc_tl_shm_tree_to_team_ranks(shm_tree->base_tree, team->base_groups[team->my_group_id].map);
    }
//    shm_tree->top_tree = NULL; //why?
    if (leaders_size > 1) {
        if (team_rank == root ||
            (root_group != team->my_group_id && UCC_SBGP_ENABLED == team->leaders_group->status)) {
            /* short cut if root is part of leaders SBGP
               Loop below is up to number of sockets/numas, ie small and fast*/
            tree_root = UCC_RANK_INVALID;
            for (int i = 0; i < team->leaders_group->group_size; i++) {
                if (ucc_ep_map_eval(team->leaders_group->map, i) == root) {
                    tree_root = i;
                    break;
                }
            }
//            size = team->leaders_group->group_size;
            rank = team->leaders_group->group_rank;
            if (tree_root != UCC_RANK_INVALID) {
                /* Root is part of leaders groop */
            	ucc_tl_shm_kn_tree_init(leaders_size, tree_root, rank, top_radix,
                            UCC_COLL_TYPE_BCAST, shm_tree->top_tree);
                /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
                ucc_tl_shm_tree_to_team_ranks(shm_tree->top_tree, team->leaders_group->map);
            } else {
                /* Build tmp ep_map for leaders + root set
                  The Leader Rank on the same base group with actual root will be replaced in the tree
                  by the root itself to save 1 extra copy in SM */
                ucc_rank_t ranks[leaders_size]; //Can be allocated on stack
                for (int i = 0; i < leaders_size; i++) {
                    ucc_rank_t leader_rank = ucc_ep_map_eval(team->leaders_group->map, i);
                    ucc_rank_t leader_group_id = ucc_ep_map_eval(team->rank_group_id_map, leader_rank);
                    if (leader_group_id == root_group) {
                        tree_root = i;
                        ucc_assert(team_rank == root);
                        ranks[i] = root;
                        rank = i;
                    } else {
                        ranks[i] = leader_rank;
                    }
                }
//                top_tree = shm_tree->top_tree;
                ucc_tl_shm_kn_tree_init(leaders_size, tree_root, rank,
                                        UCC_COLL_TYPE_BCAST, top_radix, shm_tree->top_tree);
                /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
//                shm_tree->top_tree = top_tree;
                ucc_ep_map_t map = {
                    .type  = UCC_EP_MAP_ARRAY,
                    .array.map = ranks,
                    .array.elem_size = sizeof(ucc_rank_t)
                };
                ucc_tl_shm_tree_to_team_ranks(shm_tree->top_tree, map);
            }
        }
    }
    ucc_tl_shm_cache_tree(team, base_radix, top_radix, root, UCC_COLL_TYPE_BCAST, shm_tree);
    *tree_p = shm_tree;
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_wr_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
    ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    args = TASK_ARGS(task);
    size_t             data_size = args.src.info.count *
                                   ucc_dt_size(args.src.info.datatype);
    ucc_rank_t         root = (ucc_rank_t) args.root, rank = UCC_TL_TEAM_RANK(team);
//    ucc_rank_t         group_size = UCC_TL_TEAM_SIZE(team);
    ucc_tl_shm_seg_t  *seg = task->seg;
    ucc_tl_shm_tree_t *tree = task->tree;
    int                is_inline = data_size <= team->max_inline;
    ucc_status_t       status;
    ucc_tl_shm_ctrl_t *my_ctrl, *parent_ctrl;
    void *src;

    if (rank == root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) {
            return UCC_INPROGRESS;
        }
    }
    if (tree->top_tree) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree, is_inline, rank == root,
                                data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    status = ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree, is_inline, rank == root,
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
        parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->base_tree->parent); //base_tree? was just tree->parent before
        src = is_inline ? parent_ctrl->data : ucc_tl_shm_get_data(seg, team, tree->base_tree->parent); //base_tree? was just tree->parent before
        memcpy(args.src.info.buffer, src, data_size);
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
    return UCC_OK;
}

//ucc_status_t ucc_tl_shm_bcast_ww_progress(ucc_coll_task_t *coll_task) {
//    return UCC_OK;
//}
//
//ucc_status_t ucc_tl_shm_bcast_rr_progress(ucc_coll_task_t *coll_task) {
//    return UCC_OK;
//}
//
//ucc_status_t ucc_tl_shm_bcast_rw_progress(ucc_coll_task_t *coll_task) {
//    return UCC_OK;
//}

ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
//	ucc_coll_args_t    args = TASK_ARGS(task);
    ucc_status_t status;
//    size_t data_size = args.src.info.count *
//                       ucc_dt_size(args.src.info.datatype);
//    ucc_rank_t rank = UCC_TL_TEAM_RANK(team);

    task->seq_num++;
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
	ucc_rank_t         base_radix = UCC_TL_SHM_TEAM_LIB(team)->cfg.bcast_base_radix;
	ucc_rank_t         top_radix  = UCC_TL_SHM_TEAM_LIB(team)->cfg.bcast_top_radix;
	ucc_status_t status;

    task->super.post = ucc_tl_shm_bcast_start;
    task->seq_num    = team->seq_num++;
    *task->seg       = team->segs[task->seq_num % team->n_concurrent]; //should be *task->seg = ... or teask->seg = &...?
    if (UCC_OK != (status = ucc_tl_shm_tree_init_bcast(team, args.root, base_radix, top_radix, &task->tree))) {
    	return status;
    }
//    task->super.progress = ucc_tl_shm_bcast_progress;
    switch(TASK_LIB(task)->cfg.bcast_alg) {
//        case BCAST_WW:
//        	task->super.progress = ucc_tl_shm_bcast_ww_progress;
//            break;
        case BCAST_WR:
        	task->super.progress = ucc_tl_shm_bcast_wr_progress;
            break;
//        case BCAST_RR:
//        	task->super.progress = ucc_tl_shm_bcast_rr_progress;
//            break;
//        case BCAST_RW:
//        	task->super.progress = ucc_tl_shm_bcast_rw_progress;
//            break;
    }
    return UCC_OK;
}
