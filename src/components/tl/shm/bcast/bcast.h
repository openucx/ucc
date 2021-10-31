/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef BCAST_H_
#define BCAST_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"

ucc_status_t ucc_tl_shm_bcast_init(ucc_tl_shm_task_t *task);

#endif
