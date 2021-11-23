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

enum {
    BCAST_WW,
    BCAST_WR,
    BCAST_RR,
    BCAST_RW
}; //make configurable from user for example from user "wr" to cfg->bcast_alg = 1

#endif
