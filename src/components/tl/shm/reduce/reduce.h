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
#include "components/mc/ucc_mc.h"

ucc_status_t ucc_tl_shm_reduce_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t      *tl_team,
                                    ucc_coll_task_t     **task_h);
#endif
