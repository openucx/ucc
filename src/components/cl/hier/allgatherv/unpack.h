/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

ucc_status_t ucc_cl_hier_allgatherv_unpack_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);
ucc_status_t ucc_cl_hier_allgatherv_unpack_start(ucc_coll_task_t *task);
void         ucc_cl_hier_allgatherv_unpack_progress(ucc_coll_task_t *task);
ucc_status_t ucc_cl_hier_allgatherv_unpack_finalize(ucc_coll_task_t *task);
