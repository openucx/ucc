/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_tag.h"
#include "core/ucc_mc.h"
#include "core/ucc_team.h"
#include "barrier/barrier.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "bcast/bcast.h"

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

static ucc_status_t ucc_tl_ucp_triggered_coll_complete(ucc_coll_task_t *parent_task,
                                                       ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_info(task->team->super.super.context->lib,
        "triggered collective complete. task:%p", coll_task);
    return ucc_mc_ee_task_end(coll_task->ee_task, coll_task->ee->ee_type);
}

static ucc_status_t ucc_tl_ucp_event_trigger_complete(ucc_coll_task_t *parent_task,
                                                      ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_info(task->team->super.super.context->lib, "event triggered. task:%p", coll_task);

    coll_task->ee_task = parent_task->ee_task;
    coll_task->post(coll_task);
    if (coll_task->super.status == UCC_OK) {
        return ucc_tl_ucp_triggered_coll_complete(coll_task, coll_task);
    } else {

        ucc_assert(coll_task->super.status = UCC_INPROGRESS);

        if (coll_task->ee_task) {
            ucc_event_manager_init(&coll_task->em);
            coll_task->handlers[UCC_EVENT_COMPLETED] = ucc_tl_ucp_triggered_coll_complete;
            ucc_event_manager_subscribe(&coll_task->em, UCC_EVENT_COMPLETED, coll_task);
        }
    }

    return UCC_OK;
}

//TODO can we move this logic to CORE
static ucc_status_t ucc_tl_ucp_ee_wait_for_event_trigger(ucc_coll_task_t *coll_task)
{
    ucc_ev_t *post_event;
    ucc_status_t status;
    ucc_ev_t *ev;
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->super.ev == NULL) {
        if (task->super.ee->ee_type == UCC_EE_CUDA_STREAM) {
            /* implicit event triggered */
            task->super.ev = (ucc_ev_t *) 0xFFFF; /* dummy event */
            task->super.ee_task = NULL;
        } else if (UCC_OK == ucc_ee_get_event_internal(task->super.ee, &ev,
                                                &task->super.ee->event_in_queue)) {
            tl_info(task->team->super.super.context->lib,
                    "triggered event arrivied. task:%p", coll_task);
            task->super.ev = ev;
            task->super.ee_task = NULL;
        } else {
            return UCC_OK;
        }
    }

    if (task->super.ee_task == NULL) {
        status = ucc_mc_ee_task_post(task->super.ee->ee_context,
                                     task->super.ee->ee_type, &task->super.ee_task);
        if (status != UCC_OK) {
            tl_error(task->team->super.super.context->lib, "error in ee task post");
            return status;
        }
    }

    if (task->super.ee_task == NULL ||
        (UCC_OK == ucc_mc_ee_task_query(task->super.ee_task, task->super.ee->ee_type)))
    {

        /* TODO: mpool */
        post_event = ucc_malloc(sizeof(ucc_ev_t), "event");
        if (post_event == NULL) {
            tl_error(task->team->super.super.context->lib,
                     "failed to allocate memory for event");
            return UCC_ERR_NO_MEMORY;
        }

        post_event->ev_type = UCC_EVENT_COLLECTIVE_POST;
        post_event->ev_context_size = 0;
        ucc_ee_set_event_internal(coll_task->ee, post_event, &coll_task->ee->event_out_queue);
        task->super.super.status = UCC_OK;
    }

    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_triggered_post(ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_task_t *ev_task  = ucc_tl_ucp_get_task(task->team);

    ev_task->super.ee = ee;
    ev_task->super.ev = NULL;
    ev_task->super.flags = UCC_COLL_TASK_FLAG_INTERNAL;
    ev_task->super.finalize = ucc_tl_ucp_coll_finalize;
    ev_task->super.super.status = UCC_INPROGRESS;

    ev_task->super.progress = ucc_tl_ucp_ee_wait_for_event_trigger;
    ucc_event_manager_init(&ev_task->super.em);
    coll_task->handlers[UCC_EVENT_COMPLETED] = ucc_tl_ucp_event_trigger_complete;
    ucc_event_manager_subscribe(&ev_task->super.em, UCC_EVENT_COMPLETED, coll_task);
    ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(ev_task->team)->pq, &ev_task->super);

    tl_info(task->team->super.super.context->lib, "triggered post. task:%p", coll_task);

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_tl_ucp_get_task(tl_team);
    ucc_status_t          status;

    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));
    task->team                 = tl_team;
    task->tag                  = tl_team->seq_num;
    tl_team->seq_num           = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
    task->super.finalize       = ucc_tl_ucp_coll_finalize;
    task->super.triggered_post = ucc_tl_ucp_triggered_post;
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BARRIER:
        status = ucc_tl_ucp_barrier_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_ucp_alltoall_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_ucp_alltoallv_init(task);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_ucp_allreduce_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_ucp_allgather_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        status = ucc_tl_ucp_allgatherv_init(task);
        break;
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_ucp_bcast_init(task);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    tl_info(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}
