/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "utils/ucc_coll_utils.h"
#include "allreduce/allreduce.h"

ucc_status_t ucc_cl_hier_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task)
{
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_cl_hier_allreduce_rab_init(coll_args, team, task);
    default:
        cl_error(team->context->lib, "coll_type %s is not supported",
                 ucc_coll_type_str(coll_args->args.coll_type));
        break;
    }
    return UCC_ERR_NOT_SUPPORTED;
}
