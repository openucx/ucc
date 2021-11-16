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

ucc_status_t ucc_tl_shm_bcast_init(ucc_tl_shm_task_t *task);

ucc_status_t ucc_tl_shm_tree_init_bcast(ucc_tl_shm_team_t *team,
                                        ucc_rank_t root,
                                        ucc_rank_t base_radix,
                                        ucc_rank_t top_radix,
                                        ucc_tl_shm_tree_t **tree_p);

static inline ucc_status_t ucc_tl_shm_bcast_write(ucc_tl_shm_team_t *team, ucc_tl_shm_seg_t *seg, ucc_tl_shm_task_t *task,
                                     ucc_kn_tree_t *tree, int is_inline,
                                     int is_op_root, size_t data_size)
{
	ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t seq_num = task->seq_num;
    uint32_t n_polls = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    ucc_tl_shm_ctrl_t *my_ctrl; //, *ctrl;
    void *src;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (tree->parent == UCC_RANK_INVALID) {
        /* i am root of the tree*/
        /* If the tree root is global OP root he can copy data out from origin user src buffer.
           Otherwise, it must be base_tree in 2lvl alg, and the data of the tree root is in the
           local shm (ctrl or data) */
        src = is_op_root ? task->src : (is_inline ? my_ctrl->data :
                    ucc_tl_shm_get_data(seg, team, team_rank));
        ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline, src, data_size);
    } else {
        for (int i = 0; i < n_polls; i++) {
            if (my_ctrl->pi == seq_num) {
                SHMSEG_ISYNC();
                src = is_inline ? my_ctrl->data : ucc_tl_shm_get_data(seg, team, team_rank);
                ucc_tl_shm_copy_to_children(seg, team, tree, seq_num, is_inline, src, data_size); //why group_type?
                /* copy out to user dest is done in the end of base bcast alg */
                return UCC_OK;
            }
        }
        return UCC_INPROGRESS;
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_shm_bcast_read(ucc_tl_shm_team_t *team, ucc_tl_shm_seg_t *seg, ucc_tl_shm_task_t *task,
                                    ucc_kn_tree_t *tree, int is_inline,
                                    int is_op_root, size_t data_size)
{
	ucc_rank_t team_rank = UCC_TL_TEAM_RANK(team);
    uint32_t seq_num = task->seq_num;
    uint32_t n_polls = UCC_TL_SHM_TEAM_LIB(team)->cfg.n_polls;
    void *my_data, *src, *dst;
    ucc_tl_shm_ctrl_t *parent_ctrl, *my_ctrl;

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, team_rank);

    if (is_op_root) {
        /* Only global op root needs to copy the data from user src to its shm */
        my_data = ucc_tl_shm_get_data(seg, team, team_rank);
        dst = is_inline ? my_ctrl->data : my_data;
        memcpy(dst, task->src, data_size);
        SHMSEG_WMB();
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
    } else {
        for (int i = 0; i < n_polls; i++){
            if (my_ctrl->pi == seq_num) {
                parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, tree->parent);
                SHMSEG_ISYNC();
                if (tree->n_children) {
                    src = is_inline ? parent_ctrl->data : ucc_tl_shm_get_data(seg, team, tree->parent);
                    my_data = ucc_tl_shm_get_data(seg, team, team_rank);
                    dst = is_inline ? my_ctrl->data : my_data;
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
    return UCC_OK;
}

enum {
    BCAST_WW,
    BCAST_WR,
    BCAST_RR,
    BCAST_RW
}; //make configurable from user for example from user "wr" to cfg->bcast_alg = 1

#endif
