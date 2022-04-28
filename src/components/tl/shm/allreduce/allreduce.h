/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_
#include "../tl_shm.h"
#include "../tl_shm_coll.h"
#include "../tl_shm_knomial_pattern.h"
#include "../bcast/bcast.h"
#include "../reduce/reduce.h"

ucc_status_t ucc_tl_shm_allreduce_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     team,
                                     ucc_coll_task_t **    task_h);

#endif
