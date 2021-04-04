/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_cl_basic_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task)
{
    ucc_cl_basic_team_t    *cl_team = ucc_derived_of(team, ucc_cl_basic_team_t);
    ucc_base_coll_init_fn_t init;
    ucc_base_team_t        *bteam;
    ucc_status_t            status;
    status =
        ucc_coll_score_map_lookup(cl_team->score_map, coll_args, &init, &bteam);
    if (UCC_OK != status) {
        return status;
    }
    return init(coll_args, bteam, task);
}
