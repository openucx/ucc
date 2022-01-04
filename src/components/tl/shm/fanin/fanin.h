/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef FANIN_H_
#define FANIN_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"

ucc_status_t ucc_tl_shm_fanin_init(ucc_tl_shm_task_t *task);

enum {
    FANIN_WW,
    FANIN_WR,
    FANIN_RR,
    FANIN_RW
}; //make configurable from user for example from user "wr" to cfg->fanin_alg = 1

#endif
