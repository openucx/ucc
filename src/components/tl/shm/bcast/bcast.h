/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef BCAST_H_
#define BCAST_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"

static inline ucc_status_t ucc_tl_shm_tree_init_bcast(ucc_tl_shm_team_t *team,
                                        ucc_rank_t root,
                                        ucc_rank_t base_radix,
                                        ucc_rank_t top_radix,
                                        ucc_tl_shm_tree_t **tree_p)
{
    ucc_kn_tree_t *base_tree, *top_tree;
    ucc_tl_shm_tree_t *shm_tree;
    ucc_rank_t tree_root, rank;
    size_t top_tree_size, base_tree_size;
    ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t group_size = team->base_groups[team->my_group_id].group_size;
    ucc_rank_t leaders_size = team->leaders_group->group_size;

    if (ucc_tl_shm_cache_tree_lookup(team, base_radix, top_radix, root, UCC_COLL_TYPE_BCAST, tree_p)) {
        return UCC_OK;
    }
    /* Pool is initialized using UCC_KN_TREE_SIZE macro memory estimation, using
       base_group[my_group_id]->size and max supported radix (maybe up to group size as well */
    top_tree_size = ucc_tl_shm_kn_tree_size(leaders_size, top_radix);
    base_tree_size = ucc_tl_shm_kn_tree_size(group_size, base_radix);
    shm_tree = (ucc_tl_shm_tree_t *) ucc_malloc(sizeof(ucc_kn_tree_t) * (top_tree_size + base_tree_size));//TODO: alloc_from_shm_tree_pool() ucc_mpool_get(&ctx->req_mp);

    if (!shm_tree) {
        return UCC_ERR_NO_MEMORY;
    }

    shm_tree->base_tree = shm_tree;
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
        tree_to_team_ranks(shm_tree->base_tree, team->base_groups[team->my_group_id]->map);
    }
//    shm_tree->top_tree = NULL; //why?
    if (leaders_size > 1) {
        if (team_rank == root ||
            (root_group != team->my_gruop_id && UCC_SBGP_ENABLED(team->leaders_group))) {
            ucc_assert(group_type = LEADERS_GROUP);
            /* short cut if root is part of leaders SBGP
               Loop below is up to number of sockets/numas, ie small and fast*/
            tree_root = UCC_RANK_INVALID;
            for (int i = 0; i < team->leaders_group->group_size; i++) {
                if (ucc_sbgp_rank2team(team->leaders_group, i) == root) {
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
                tree_to_team_ranks(&shm_tree->top_tree, team->leaders_group->map);
            } else {
                /* Build tmp ep_map for leaders + root set
                  The Leader Rank on the same base group with actual root will be replaced in the tree
                  by the root itself to save 1 extra copy in SM */
                ucc_rank_t ranks[leaders_size]; //Can be allocated on stack
                for (int i = 0; i < leaders_size; i++) {
                    ucc_rank_t leader_rank = ucc_sbgp_rank2team(team->leaders_group, i);
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
                    .array = ranks
                    };
                tree_to_team_ranks(shm_tree->top_tree, map);
            }
        }
    }
    ucc_tl_shm_cache_tree(team, base_radix, top_radix, root, UCC_COLL_TYPE_BCAST, shm_tree);
    *tree_p = shm_tree;
    return UCC_OK;
}

static inline ucc_tl_shm_bcast_write(ucc_tl_shm_seg_t *seg, ucc_tl_shm_task_t *task,
                                     ucc_kn_tree_t *tree, int is_inline,
                                     int is_op_root, size_t data_size)
{
	ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t seq_num = task->seq_num;
    uint32_t n_polls = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls; // what should n_polls default be?
    ucc_tl_shm_ctrl_t *my_ctrl; //, *ctrl;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->parent == UCC_RANK_INVALID) {
        /* i am root of the tree*/
        /* If the tree root is global OP root he can copy data out from origin user src buffer.
           Otherwise, it must be base_tree in 2lvl alg, and the data of the tree root is in the
           local shm (ctrl or data) */
        void *src = is_op_root ? task->src : (is_inline ? my_ctrl->data :
                                                get_data(seg, team, team_rank));
        copy_to_children(seg, team, tree, is_inline, src, data_size);
    } else {
        for (int i = 0; i < n_polls; i++) {
            if (my_ctrl->pi == seq_num) {
                SHMSEG_ISYNC();
                void *src = is_inline ? my_ctrl->data : get_data(seg, team, team_rank);
                copy_to_children(seg, team, tree, group_type, is_inline, src, data_size);
                /* copy out to user dest is done in the end of base bcast alg */
                return UCC_OK;
            }
        }
        return UCC_INPROGRESS;
    }
    return UCC_OK;
}

static inline ucc_tl_shm_bcast_read(ucc_tl_shm_seg_t *seg, ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    int is_op_root, size_t data_size)
{
	ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t seq_num = task->seq_num;
    uint32_t n_polls = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void *my_data;
    ucc_tl_shm_ctrl_t *parent_ctrl, *my_ctrl;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (is_op_root) {
        /* Only global op root needs to copy the data from user src to its shm */
        memcpy(dst, task->src, data_size);
        SHMSEG_WMB();
        signal_to_children(seg, team, seq_num, tree);
    } else {
        for (int i = 0; i < n_polls; i++){
            if (my_ctrl->pi == seq_num) {
                parent_ctrl = get_ctrl(seg, team, tree->parent);
                SHMSEG_ISYNC();
                if (tree->n_children) {
                    void *src = is_inline ? parent_ctrl->data : get_data(seg, team, tree->parent);
                    my_data = get_data(seg, team, team_rank);
                    void *dst = is_inline ? my_ctrl->data : my_data;
                    memcpy(dst, src, data_size);
                    SHMSEG_WMB();
                    signal_to_children(seg, team, seq_num, tree);
                }
                /* copy out to user dest is done in the end of base bcast alg */
                return UCC_OK;
            }
        }
        return UCC_INPROGRESS;
    }
    return UCC_OK;
}

enum {
    BCAST_WW,
    BCAST_WR,
    BCAST_RR,
    BCAST_RW
}; //make configurable from user for example from user "wr" to cfg->bcast_alg = 1

ucc_status_t ucc_tl_shm_bcast_init(ucc_tl_shm_task_t *task);

#endif
