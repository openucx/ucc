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
    } else {
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
    return UCC_OK;
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
        //TODO there will be 2 copies for RR case: need to fix
        /* Only global op root needs to copy the data from user src to its shm */
        if (*is_op_root == 1) {
            dst = is_inline ? my_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  team_rank);
            memcpy(dst, TASK_ARGS(task).src.info.buffer, data_size);
            SHMSEG_WMB();
            (*is_op_root)++;
        }
        ucc_tl_shm_signal_to_children(seg, team, seq_num, tree);
    } else {
        parent = tree->parent;
        if (parent == UCC_RANK_INVALID) {
            /* I'm the root of the tree and NOT is_op_root. It means the tree is
               base tree and i already have the data in my shm via top_tree step
               (read or write). Just notify children. */
            ucc_assert(task->tree->base_tree == tree);
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
    return UCC_OK;
}
ucc_status_t ucc_tl_shm_tree_init_bcast(ucc_tl_shm_team_t *team,
                                        ucc_rank_t root,
                                        ucc_rank_t base_radix,
                                        ucc_rank_t top_radix,
                                        int *tree_in_cache,
                                        ucc_tl_shm_tree_t **tree_p)
{
    ucc_kn_tree_t *base_tree, *top_tree;
    ucc_rank_t     tree_root, rank, local_rank, leader_rank, leader_group_id;
    size_t         top_tree_size, base_tree_size;
    int i;

    ucc_rank_t  team_rank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t  group_size   = team->base_groups[team->my_group_id].group_size;
    ucc_rank_t  leaders_size = team->leaders_group->group_size;
    ucc_rank_t  root_group   = ucc_ep_map_eval(team->rank_group_id_map, root);
    ucc_sbgp_t *sbgp         = &team->base_groups[team->my_group_id];

    ucc_tl_shm_tree_t *shm_tree = (ucc_tl_shm_tree_t *)
                                  ucc_malloc(sizeof(ucc_kn_tree_t *) * 2);

    if (ucc_tl_shm_cache_tree_lookup(team, base_radix, top_radix, root,
                                     UCC_COLL_TYPE_BCAST, tree_p) == 1) {
    	*tree_in_cache = 1;
        return UCC_OK;
    }
    /* Pool is initialized using UCC_KN_TREE_SIZE macro memory estimation, using
       base_group[my_group_id]->size and max supported radix (maybe up to group size as well */
    top_tree_size = ucc_tl_shm_kn_tree_size(leaders_size, top_radix);
    base_tree_size = ucc_tl_shm_kn_tree_size(group_size, base_radix);
    base_tree = (ucc_kn_tree_t *) ucc_malloc(sizeof(ucc_rank_t) *
                                             (base_tree_size + 2));

    if (!base_tree) {
        return UCC_ERR_NO_MEMORY;
    }

    top_tree = (ucc_kn_tree_t *) ucc_malloc(sizeof(ucc_rank_t) *
                                            (top_tree_size + 2));

    if (!top_tree) {
    	ucc_free(base_tree);
        return UCC_ERR_NO_MEMORY;
    }

    shm_tree->base_tree = NULL;
    shm_tree->top_tree = NULL;
    for (i = 0; i < sbgp->group_size; i++) {
    	if (ucc_ep_map_eval(sbgp->map, i) == team_rank) {
    		local_rank = i;
    		break;
    	}
    }
    ucc_assert(i < sbgp->group_size);

    if (group_size > 1) { //should be >= in case of only np=2 (in total)?
        rank = local_rank;
        if (team->my_group_id == root_group) {
            tree_root = ucc_ep_map_eval(team->group_rank_map, root);
        } else {
            tree_root = 0;
        }
        ucc_tl_shm_kn_tree_init(group_size, tree_root, rank, base_radix,
                                UCC_COLL_TYPE_BCAST, base_tree);
        shm_tree->base_tree = base_tree;
        /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
        ucc_tl_shm_tree_to_team_ranks(base_tree,
                                      team->base_groups[team->my_group_id].map);
    }
    if (leaders_size > 1) {
        if (team_rank == root ||
            (root_group != team->my_group_id &&
             UCC_SBGP_ENABLED == team->leaders_group->status)) {
            /* short cut if root is part of leaders SBGP
               Loop below is up to number of sockets/numas, ie small and fast*/
            tree_root = UCC_RANK_INVALID;
            for (i = 0; i < team->leaders_group->group_size; i++) {
                if (ucc_ep_map_eval(team->leaders_group->map, i) == root) {
                    tree_root = i;
                    break;
                }
            }
            rank = team->leaders_group->group_rank;
            if (tree_root != UCC_RANK_INVALID) {
                /* Root is part of leaders groop */
            	ucc_tl_shm_kn_tree_init(leaders_size, tree_root, rank,
                                        top_radix, UCC_COLL_TYPE_BCAST,
                                        top_tree);
                /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
                ucc_tl_shm_tree_to_team_ranks(top_tree,
                                              team->leaders_group->map);
            } else {
                /* Build tmp ep_map for leaders + root set
                  The Leader Rank on the same base group with actual root will be replaced in the tree
                  by the root itself to save 1 extra copy in SM */
                ucc_rank_t ranks[leaders_size]; //Can be allocated on stack
                for (i = 0; i < leaders_size; i++) {
                    leader_rank = ucc_ep_map_eval(team->leaders_group->map, i);
                    leader_group_id = ucc_ep_map_eval(team->rank_group_id_map,
                                                      leader_rank);
                    if (leader_group_id == root_group) {
                        tree_root = i;
                        ranks[i] = root;
                        if (team_rank == root) {
                            rank = i;
                        }
                    } else {
                        ranks[i] = leader_rank;
                    }
                }
                ucc_tl_shm_kn_tree_init(leaders_size, tree_root, rank,
                                        UCC_COLL_TYPE_BCAST, top_radix,
                                        top_tree);
                /* Convert the tree to origin TL/TEAM ranks from the BASE_GROUP ranks*/
                ucc_ep_map_t map = {
                    .type  = UCC_EP_MAP_ARRAY,
                    .array.map = ranks,
                    .array.elem_size = sizeof(ucc_rank_t)
                };
                ucc_tl_shm_tree_to_team_ranks(top_tree, map);
            }
            shm_tree->top_tree = top_tree;
        }
    }
    *tree_in_cache = ucc_tl_shm_cache_tree(team, base_radix, top_radix, root,
                                           UCC_COLL_TYPE_BCAST, shm_tree);
    *tree_p = shm_tree;
    return UCC_OK;
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

    if (is_op_root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) { //TODO: implement
            return UCC_INPROGRESS;
        }
    }
    if (tree->top_tree) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree,
                                        is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    status = ucc_tl_shm_bcast_write(team, seg, task, tree->base_tree,
                                    is_inline, &is_op_root, data_size);

    if (UCC_OK != status) {
        /* in progress */
        return status;
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

    if (is_op_root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) { //TODO: implement
            return UCC_INPROGRESS;
        }
    }

    if (tree->top_tree) {
        status = ucc_tl_shm_bcast_write(team, seg, task, tree->top_tree,
                                        is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    status = ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree, is_inline,
                                   &is_op_root, data_size);

    if (UCC_OK != status) {
        /* in progress */
        return status;
    }

    /* Copy out to user dest:
       - ranks that are non-roots in the base tree (those that have parent)
         will have the data in their parent's data/ctrl because we did READ
         on the last step.
       - other ranks must be participants of the top tree WRITE step, hence
        they have the data in their local shm data/ctrl */
    if (!is_op_root) {
        parent = tree->base_tree->parent == UCC_RANK_INVALID ?
                                            rank : tree->base_tree->parent;
        parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
        src = is_inline ? parent_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  parent);
        memcpy(args.src.info.buffer, src, data_size);
    }
    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
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

    if (is_op_root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) { //TODO: implement
            return UCC_INPROGRESS;
        }
    }

    if (tree->top_tree) {
        status = ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                                       is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }
    status = ucc_tl_shm_bcast_read(team, seg, task, tree->base_tree,
                                   is_inline, &is_op_root, data_size);
    if (UCC_OK != status) {
        /* in progress */
        return status;
    }

    /* Copy out to user dest:
       Where is data?
       If we did READ as 2nd step then the data is in the base_tree->parent SHM
       If we did WRITE as 2nd step then the data is in my SHM */
    if (!is_op_root) {
        parent = tree->base_tree->parent == UCC_RANK_INVALID ?
                 tree->top_tree->parent : tree->base_tree->parent;
        parent_ctrl = ucc_tl_shm_get_ctrl(seg, team, parent);
        src = is_inline ? parent_ctrl->data : ucc_tl_shm_get_data(seg, team,
                                                                  parent);
        memcpy(args.src.info.buffer, src, data_size);
    }

    my_ctrl = ucc_tl_shm_get_ctrl(seg, team, rank);
    my_ctrl->ci = task->seq_num;
    /* bcast done */
    task->super.super.status = UCC_OK;
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

    if (is_op_root) {
        /* checks if previous collective has completed on the seg
           TODO: can be optimized if we detect bcast->reduce pattern.*/
        if (UCC_OK != ucc_tl_shm_seg_ready(seg)) { //TODO: implement
            return UCC_INPROGRESS;
        }
    }
    if (tree->top_tree) {
        status = ucc_tl_shm_bcast_read(team, seg, task, tree->top_tree,
                                       is_inline, &is_op_root, data_size);
        if (UCC_OK != status) {
            /* in progress */
            return status;
        }
    }

    status = ucc_tl_shm_bcast_write(team, seg, task, tree->base_tree,
                                    is_inline, &is_op_root, data_size);

    if (UCC_OK != status) {
        /* in progress */
        return status;
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
    return UCC_OK;
}

ucc_status_t ucc_tl_shm_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_shm_task_t *task = ucc_derived_of(coll_task, ucc_tl_shm_task_t);
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
    ucc_status_t       status;

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
	ucc_tl_shm_team_t *team = TASK_TEAM(task);
	ucc_coll_args_t    args = TASK_ARGS(task);
	ucc_rank_t   base_radix = UCC_TL_SHM_TEAM_LIB(team)->cfg.bcast_base_radix;
	ucc_rank_t   top_radix  = UCC_TL_SHM_TEAM_LIB(team)->cfg.bcast_top_radix;
	ucc_status_t status;

    task->super.post = ucc_tl_shm_bcast_start;
    task->seq_num    = team->seq_num++;
    task->seg        = &team->segs[task->seq_num % team->n_concurrent];
    status = ucc_tl_shm_tree_init_bcast(team, args.root, base_radix, top_radix,
                                        &task->tree_in_cache, &task->tree);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to init shm tree");
    	return status;
    }

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
