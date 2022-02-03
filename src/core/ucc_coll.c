/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team.h"
#include "ucc_context.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "components/cl/ucc_cl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_time.h"
#include "utils/profile/ucc_profile_core.h"
#include "schedule/ucc_schedule.h"
#include "coll_score/ucc_coll_score.h"
#include "ucc_ee.h"

#define UCC_BUFFER_INFO_CHECK_MEM_TYPE(_info) do {                             \
    if ((_info).mem_type == UCC_MEMORY_TYPE_UNKNOWN) {                         \
        ucc_mem_attr_t mem_attr;                                               \
        ucc_status_t st;                                                       \
        mem_attr.field_mask = UCC_MEM_ATTR_FIELD_MEM_TYPE;                     \
        st = ucc_mc_get_mem_attr((_info).buffer, &mem_attr);                   \
        if (ucc_unlikely(st != UCC_OK)) {                                      \
            return st;                                                         \
        }                                                                      \
        (_info).mem_type = mem_attr.mem_type;                                  \
    }                                                                          \
} while(0)

#define UCC_BUFFER_INFO_CHECK_DATATYPE(_info1, _info2) do {                    \
    if ((_info1).datatype != (_info2).datatype) {                              \
        ucc_error("datatype missmatch");                                       \
        return UCC_ERR_INVALID_PARAM;                                          \
    }                                                                          \
} while(0)

#define UCC_IS_ROOT(_args, _myrank) ((_args).root == (_myrank))

#if ENABLE_DEBUG == 1
static ucc_status_t ucc_check_coll_args(const ucc_coll_args_t *coll_args,
                                        ucc_rank_t rank)
{
    switch (coll_args->coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_DATATYPE(coll_args->src.info,
                                           coll_args->dst.info);
        }
        break;
    case UCC_COLL_TYPE_REDUCE:
        if (!UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_DATATYPE(coll_args->src.info,
                                           coll_args->dst.info);
        }
        break;
    default:
        return UCC_OK;
    }
    return UCC_OK;
}
#else
#define ucc_check_coll_args(...) UCC_OK
#endif

static ucc_status_t ucc_coll_args_check_mem_type(ucc_coll_args_t *coll_args,
                                                 ucc_rank_t rank)
{
    switch (coll_args->coll_type) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return UCC_OK;
    case UCC_COLL_TYPE_BCAST:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        return UCC_OK;
    case UCC_COLL_TYPE_ALLREDUCE:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        } else {
            coll_args->src.info.mem_type = coll_args->dst.info.mem_type;
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_ALLTOALLV:
        UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        if (!UCC_IS_INPLACE(*coll_args)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info_v);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_GATHER:
    	// TODO: Check logic for Gather once implemented
    case UCC_COLL_TYPE_REDUCE:
        if (!UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        } else {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
            if (!UCC_IS_INPLACE(*coll_args)) {
                UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        	}
        }
        return UCC_OK;
    case UCC_COLL_TYPE_GATHERV:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info_v);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_SCATTER:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        }
        return UCC_OK;
    case UCC_COLL_TYPE_SCATTERV:
        if (UCC_IS_ROOT(*coll_args, rank)) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->src.info_v);
        }
        if (!(UCC_IS_INPLACE(*coll_args) && UCC_IS_ROOT(*coll_args, rank))) {
            UCC_BUFFER_INFO_CHECK_MEM_TYPE(coll_args->dst.info);
        }
        return UCC_OK;
    default:
        ucc_error("unknown collective type");
        return UCC_ERR_INVALID_PARAM;
    };
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_init,
                      (coll_args, request, team), ucc_coll_args_t *coll_args,
                      ucc_coll_req_h *request, ucc_team_h team)
{
    ucc_coll_task_t       *task;
    ucc_base_coll_args_t   op_args;
    ucc_status_t           status;

    status = ucc_coll_args_check_mem_type(coll_args, team->rank);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("memory type detection failed");
        return status;
    }
    status = ucc_check_coll_args(coll_args, team->rank);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("collective arguments check failed");
        return status;
    }

    /* TO discuss: maybe we want to pass around user pointer ? */
    op_args.mask = 0;
    memcpy(&op_args.args, coll_args, sizeof(ucc_coll_args_t));
    op_args.team = team;

    op_args.args.flags = 0;
    UCC_COPY_PARAM_BY_FIELD(&op_args.args, coll_args, UCC_COLL_ARGS_FIELD_FLAGS,
                            flags);

    status = ucc_coll_init(team->score_map, &op_args, &task);

    if (UCC_ERR_NOT_SUPPORTED == status) {
        ucc_debug("failed to init collective: not supported");
        return status;
    } else if (ucc_unlikely(status < 0)) {
        ucc_error("failed to init collective: %s", ucc_status_string(status));
        return status;
    }
    if (coll_args->mask & UCC_COLL_ARGS_FIELD_CB) {
        task->cb = coll_args->cb;
        task->flags |= UCC_COLL_TASK_FLAG_CB;
    }
    task->seq_num = team->seq_num++;
    if (ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_DEBUG) {
        char coll_debug_str[256];
        ucc_coll_str(task, coll_debug_str, sizeof(coll_debug_str));
        ucc_debug("coll_init: %s", coll_debug_str);
    }
    *request = &task->super;
    return UCC_OK;
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_post, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    ucc_debug("coll_post: req %p, seq_num %u", task, task->seq_num);

    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        task->start_time = ucc_get_time();
    }

    return task->post(task);
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_finalize, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    ucc_debug("coll_finalize: req %p, seq_num %u", task, task->seq_num);
    return task->finalize(task);
}

static ucc_status_t ucc_triggered_task_finalize(ucc_coll_task_t *task)
{
    ucc_trace("finalizing triggered ev task %p", task);
    ucc_free(task);
    return UCC_OK;
}

static ucc_status_t ucc_triggered_coll_complete(ucc_coll_task_t *parent_task, //NOLINT
                                                ucc_coll_task_t *task)
{
    ucc_trace("triggered collective complete, task %p, seq_num %u",
              task, task->seq_num);
    return ucc_ec_task_end(task->ee_task, task->ee->ee_type);
}

static ucc_status_t ucc_trigger_complete(ucc_coll_task_t *parent_task,
                                         ucc_coll_task_t *task)
{
    ucc_status_t status;

    ucc_trace("event triggered, ev_task %p, coll_task %p, seq_num %u",
              parent_task, task, task->seq_num);

    task->ee_task = parent_task->ee_task;
    status        = task->post(task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to post triggered coll, task %p, seq_num %u, %s",
                  task, task->seq_num, ucc_status_string(status));
        return status;
    }

    if (task->super.status == UCC_OK) {
        return ucc_triggered_coll_complete(task, task);
    } else {
        ucc_assert(task->super.status == UCC_INPROGRESS);
        if (task->ee_task) {
            // TODO use CB instead of EM
            ucc_event_manager_subscribe(&task->em, UCC_EVENT_COMPLETED, task,
                                        ucc_triggered_coll_complete);
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_trigger_test(ucc_coll_task_t *task)
{
    ucc_status_t status;
    ucc_ev_t     post_event;
    ucc_ev_t    *ev;

    if (task->ev == NULL) {
        if (task->ee->ee_type == UCC_EE_CUDA_STREAM) {
            /* implicit event triggered */
            task->ev = (ucc_ev_t *) 0xFFFF; /* dummy event */
            task->ee_task = NULL;
        } else if (UCC_OK == ucc_ee_get_event_internal(task->ee, &ev,
                                                 &task->ee->event_in_queue)) {
            ucc_trace("triggered event arrived, ev_task %p", task);
            task->ev      = ev;
            task->ee_task = NULL;
        } else {
            return UCC_OK;
        }
    }

    if (task->ee_task == NULL) {
        status = ucc_ec_task_post(task->ee->ee_context, task->ee->ee_type,
                                  &task->ee_task);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("error in ee task post, %s", ucc_status_string(status));
            task->super.status = status;
            return status;
        }

        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.req             = &task->triggered_task->super;
        ucc_ee_set_event_internal(task->ee, &post_event,
                                  &task->ee->event_out_queue);
    }

    if (task->ee_task == NULL ||
        (UCC_OK == ucc_ec_task_query(task->ee_task, task->ee->ee_type))) {
        task->super.status = UCC_OK;
    }
    return task->super.status;
}

ucc_status_t ucc_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                ucc_coll_task_t *task)
{
    ucc_coll_task_t *ev_task;
    ucc_status_t     status;

    if (ev->ev_type != UCC_EVENT_COMPUTE_COMPLETE) {
        ucc_error("event type %d is not supported", ev->ev_type);
        return UCC_ERR_NOT_IMPLEMENTED;
    }
    task->ee = ee;
    ev_task = ucc_malloc(sizeof(*ev_task), "ev_task");
    if (!ev_task) {
        ucc_error("failed to allocate %zd bytes for ev_task",
                  sizeof(*ev_task));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_coll_task_init(ev_task, NULL, task->team);
    ev_task->ee             = ee;
    ev_task->ev             = NULL;
    ev_task->triggered_task = task;
    ev_task->flags          = UCC_COLL_TASK_FLAG_INTERNAL;
    ev_task->finalize       = ucc_triggered_task_finalize;
    ev_task->progress       = ucc_trigger_test;
    ev_task->super.status   = UCC_INPROGRESS;

    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        UCC_COLL_SET_TIMEOUT(ev_task, task->bargs.args.timeout);
    }
    ucc_event_manager_subscribe(&ev_task->em, UCC_EVENT_COMPLETED, task,
                                ucc_trigger_complete);

    status = ucc_trigger_test(ev_task);
    if (ucc_unlikely(status < 0)) {
        ucc_free(ev_task);
        task->super.status = status;
        ucc_task_complete(task);
        return status;
    }

    if (ev_task->super.status == UCC_OK) {
        ucc_trigger_complete(ev_task, task);
        ucc_free(ev_task);
    } else {
        ucc_progress_enqueue(UCC_TASK_CORE_CTX(ev_task)->pq, ev_task);
    }
    return UCC_OK;
}

ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t *ev)
{
    ucc_coll_task_t *task = ucc_derived_of(ev->req, ucc_coll_task_t);

    ucc_debug("triggered_post: task %p, seq_num %u", task, task->seq_num);

    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        task->start_time = ucc_get_time();
    }
    return task->triggered_post(ee, ev, task);
}
