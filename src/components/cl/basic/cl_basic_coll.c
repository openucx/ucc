/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"

ucc_status_t ucc_cl_basic_coll_init(ucc_base_coll_op_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task)
{
    ucc_cl_basic_team_t *cl_team = ucc_derived_of(team, ucc_cl_basic_team_t);
    return UCC_TL_TEAM_IFACE(cl_team->tl_ucp_team)
        ->coll.init(coll_args, &cl_team->tl_ucp_team->super, task);
}
