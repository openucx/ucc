/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef BARRIER_H_
#define BARRIER_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"
#include "../fanin/fanin.h"
#include "../fanout/fanout.h"

ucc_status_t ucc_tl_shm_barrier_init(ucc_tl_shm_task_t *task);

#endif
