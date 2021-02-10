/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_context.h"
#include "components/cl/ucc_cl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "schedule/ucc_schedule.h"

/* NOLINTNEXTLINE  */
static ucc_cl_team_t *ucc_select_cl_team(ucc_coll_op_args_t *coll_args,
                                         ucc_team_t *team)
{
    /* TODO1: collective CL selection logic will be there.
       for now just return 1st CL on a list
       TODO2: remove NOLINT once TODO1 is done */
    return team->cl_teams[0];
}

ucc_status_t ucc_collective_init(ucc_coll_op_args_t *coll_args,
                                 ucc_coll_req_h *request, ucc_team_h team)
{
    ucc_cl_team_t          *cl_team;
    ucc_base_coll_op_args_t op_args;
    ucc_status_t            status;
    ucc_coll_task_t        *task;
    /* TO discuss: maybe we want to pass around user pointer ? */
    memcpy(&op_args.args, coll_args, sizeof(ucc_coll_op_args_t));
    cl_team = ucc_select_cl_team(coll_args, team);
    status =
        UCC_CL_TEAM_IFACE(cl_team)->coll.init(&op_args, &cl_team->super, &task);
    if (status != UCC_OK) {
        //TODO more descriptive error msg
        ucc_error("failed to init collective");
        return status;
    }
    *request = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_collective_post(ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);
    return task->post(request);
}

ucc_status_t ucc_collective_finalize(ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);
    return task->finalize(request);
}
