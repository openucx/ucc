/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef BARRIER_H_
#define BARRIER_H_
#include "../cl_hier.h"

ucc_status_t ucc_cl_hier_barrier_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_coll_task_t     **task);

#endif
