/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_team.h"
#include "barrier/barrier.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "bcast/bcast.h"
const char
    *ucc_tl_ucp_default_alg_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR] = {
        UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR};

void ucc_tl_ucp_send_completion_cb(void *request, ucs_status_t status,
                                   void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
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
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(task->team->super.super.context->lib,
                 "failure in send completion %s", ucs_status_string(status));
        task->super.super.status = ucs_status_to_ucc_status(status);
    }
    task->recv_completed++;
    ucp_request_free(request);
}

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    tl_info(task->team->super.super.context->lib, "finalizing ev_task %p",
            task);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

static ucc_status_t
ucc_tl_ucp_triggered_coll_complete(ucc_coll_task_t *parent_task, //NOLINT
                                   ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_info(task->team->super.super.context->lib,
        "triggered collective complete. task:%p", coll_task);
    return ucc_mc_ee_task_end(coll_task->ee_task, coll_task->ee->ee_type);
}

static ucc_status_t
ucc_tl_ucp_event_trigger_complete(ucc_coll_task_t *parent_task,
                                  ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t status;

    tl_info(task->team->super.super.context->lib,
            "event triggered. ev_task:%p coll_task:%p", parent_task, coll_task);

    coll_task->ee_task = parent_task->ee_task;
    status = coll_task->post(coll_task);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(task->team->super.super.context->lib,
                 "Failed post the triggered collecitve. task:%p", coll_task);
        return status;
    }

    if (coll_task->super.status == UCC_OK) {
        return ucc_tl_ucp_triggered_coll_complete(coll_task, coll_task);
    } else {
        ucc_assert(coll_task->super.status == UCC_INPROGRESS);
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
                    "triggered event arrived. ev_task:%p", coll_task);
            task->super.ev = ev;
            task->super.ee_task = NULL;
        } else {
            return UCC_OK;
        }
    }

    if (task->super.ee_task == NULL) {
        status = ucc_mc_ee_task_post(task->super.ee->ee_context,
                                     task->super.ee->ee_type, &task->super.ee_task);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(task->team->super.super.context->lib, "error in ee task post");
            task->super.super.status = status;
            return status;
        }

        /* TODO: mpool */
        post_event = ucc_malloc(sizeof(ucc_ev_t), "event");
        if (ucc_unlikely(post_event == NULL)) {
            tl_error(task->team->super.super.context->lib,
                     "failed to allocate memory for event");
            return UCC_ERR_NO_MEMORY;
        }

        post_event->ev_type = UCC_EVENT_COLLECTIVE_POST;
        post_event->ev_context_size = 0;
        post_event->req = &coll_task->triggered_task->super;
        ucc_ee_set_event_internal(coll_task->ee, post_event, &coll_task->ee->event_out_queue);
    }

    if (task->super.ee_task == NULL ||
        (UCC_OK == ucc_mc_ee_task_query(task->super.ee_task, task->super.ee->ee_type)))
    {
        task->super.super.status = UCC_OK;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_triggered_post(ucc_ee_h ee, ucc_ev_t *ev, //NOLINT
                                       ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_task_t *ev_task = ucc_tl_ucp_get_task(task->team);
    ucc_status_t       status;

    ucc_coll_task_init(&ev_task->super);
    ev_task->super.ee             = ee;
    ev_task->super.ev             = NULL;
    ev_task->super.triggered_task = coll_task;
    ev_task->super.flags          = UCC_COLL_TASK_FLAG_INTERNAL;
    ev_task->super.finalize       = ucc_tl_ucp_coll_finalize;
    ev_task->super.super.status   = UCC_INPROGRESS;

    tl_info(task->team->super.super.context->lib,
            "triggered post. ev_task:%p coll_task:%p", &ev_task->super, coll_task);
    ev_task->super.progress = ucc_tl_ucp_ee_wait_for_event_trigger;
    ucc_event_manager_init(&ev_task->super.em);
    coll_task->handlers[UCC_EVENT_COMPLETED] = ucc_tl_ucp_event_trigger_complete;
    ucc_event_manager_subscribe(&ev_task->super.em, UCC_EVENT_COMPLETED, coll_task);

    status = ucc_tl_ucp_ee_wait_for_event_trigger(&ev_task->super);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    if (ev_task->super.super.status == UCC_OK) {
        ucc_tl_ucp_event_trigger_complete(&ev_task->super, coll_task);
        ucc_tl_ucp_put_task(ev_task);
        return UCC_OK;
    }
    ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(ev_task->team)->pq, &ev_task->super);

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t    *task = ucc_tl_ucp_init_task(coll_args, team);
    ucc_status_t          status;

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
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    tl_info(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}

static inline int alg_id_from_str(ucc_coll_type_t coll_type, const char *str)
{
    switch (coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_ucp_allreduce_alg_from_str(str);
    default:
        break;
    }
    return -1;
}

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t   coll_type,
                                       ucc_memory_type_t mem_type, //NOLINT
                                       ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    if (alg_id_str) {
        alg_id = alg_id_from_str(coll_type, alg_id_str);
    }

    switch (coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        switch (alg_id) {
        case UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_knomial_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_sra_knomial_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
