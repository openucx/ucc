/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_H_
#define REDUCE_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "core/ucc_mc.h"

ucc_status_t ucc_tl_shm_reduce_init(ucc_tl_shm_task_t *task);

#endif
