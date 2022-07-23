/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_cl_basic_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task)
{
    ucc_cl_basic_team_t *cl_team = ucc_derived_of(team, ucc_cl_basic_team_t);
    ucc_status_t         status;

    status = ucc_coll_init(cl_team->score_map, coll_args, task);
    if (UCC_ERR_NOT_FOUND == status) {
        cl_warn(UCC_CL_TEAM_LIB(cl_team),
                "no TL supporting given coll args is available");
        return UCC_ERR_NOT_SUPPORTED;
    }
    return status;
}
