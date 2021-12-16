/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_H_
#define REDUCE_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"
#include "core/ucc_mc.h"

ucc_status_t ucc_tl_shm_reduce_init(ucc_tl_shm_task_t *task);

enum {
    REDUCE_WW,
    REDUCE_WR,
    REDUCE_RR,
    REDUCE_RW
}; //make configurable from user for example from user "wr" to cfg->bcast_alg = 1

#endif
