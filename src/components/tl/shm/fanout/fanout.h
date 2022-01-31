/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef FANOUT_H_
#define FANOUT_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"


ucc_status_t ucc_tl_shm_fanout_signal(ucc_tl_shm_team_t *team,
                                      ucc_tl_shm_seg_t *seg,
                                      ucc_tl_shm_task_t *task,
                                      ucc_kn_tree_t *tree,
                                      int is_op_root);

ucc_status_t ucc_tl_shm_fanin_init(ucc_tl_shm_task_t *task);

#endif
