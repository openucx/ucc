/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"

static ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_req_t *request)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(request, ucc_tl_ucp_task_t);
    tl_info(task->team->super.super.context->lib, "finalizing coll req %p",
            request);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

static ucc_status_t ucc_tl_ucp_coll_post(ucc_coll_req_t *request)
{
    /*This is an example fn, the actual post fn for a request should be set
      during init */
    ucc_tl_ucp_task_t *task = ucc_derived_of(request, ucc_tl_ucp_task_t);
    tl_info(task->team->super.super.context->lib, "posting coll req %p",
            request);
    request->status = UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_op_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_task_t    *task    = ucc_tl_ucp_get_task(ctx);
    task->team                    = tl_team;
    task->super.super.post        = ucc_tl_ucp_coll_post;
    task->super.super.finalize    = ucc_tl_ucp_coll_finalize;
    *task_h                       = &task->super;
    return UCC_OK;
}
