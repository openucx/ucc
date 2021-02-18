/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_tag.h"
#include "barrier/barrier.h"

void ucc_tl_ucp_send_completion_cb(void *request, ucs_status_t status,
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (UCS_OK != status) {
        tl_error(task->team->super.super.context->lib,
                 "failure in send completion %s", ucs_status_string(status));
        task->super.super.status = ucs_status_to_ucc_status(status);
    }
    task->send_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb(void *request, ucs_status_t status,
                                   const ucp_tag_recv_info_t *info, /* NOLINT */
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (UCS_OK != status) {
        tl_error(task->team->super.super.context->lib,
                 "failure in send completion %s", ucs_status_string(status));
        task->super.super.status = ucs_status_to_ucc_status(status);
    }
    task->recv_completed++;
    ucp_request_free(request);
}

static ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    tl_info(task->team->super.super.context->lib, "finalizing coll task %p",
            task);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_op_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_tl_ucp_get_task(tl_team);
    ucc_status_t          status;
    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_op_args_t));
    task->team           = tl_team;
    task->tag            = tl_team->seq_num++; //TODO Wrap around over max tag
    task->super.finalize = ucc_tl_ucp_coll_finalize;
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BARRIER:
        status = ucc_tl_ucp_barrier_init(task);
        break;
    default:
        ucc_tl_ucp_put_task(task);
        return UCC_ERR_NOT_SUPPORTED;
    }
    tl_info(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}
