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
    ucc_coll_task_t          *task;
    ucc_base_coll_args_t      op_args;
    ucc_status_t              status;
    ucc_ee_executor_params_t  params;
    ucc_memory_type_t         coll_mem_type;
    ucc_ee_type_t             coll_ee_type;

    /* Global check to reduce the amount of checks throughout
       all TLs */
    if (UCC_COLL_ARGS_ACTIVE_SET(coll_args) &&
        ((UCC_COLL_TYPE_BCAST != coll_args->coll_type) ||
         coll_args->active_set.size != 2)) {
        ucc_warn("Active Sets are only supported for bcast and set size = 2");
        return UCC_ERR_NOT_SUPPORTED;
    }

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

    task->flags |= UCC_COLL_TASK_FLAG_TOP_LEVEL;
    if (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
        task->flags |= UCC_COLL_TASK_FLAG_EXECUTOR_STOP;
        coll_mem_type = ucc_coll_args_mem_type(&op_args);
        switch(coll_mem_type) {
        case UCC_MEMORY_TYPE_CUDA:
            coll_ee_type = UCC_EE_CUDA_STREAM;
            break; 
        case UCC_MEMORY_TYPE_ROCM:
            coll_ee_type = UCC_EE_ROCM_STREAM;
            break;
       case UCC_MEMORY_TYPE_HOST:
            coll_ee_type = UCC_EE_CPU_THREAD;
            break;
        default:
            ucc_error("no suitable executor available for memory type %s",
                      ucc_memory_type_names[coll_mem_type]);
            status = UCC_ERR_INVALID_PARAM;
            goto coll_finalize;
        }
        params.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
        params.ee_type = coll_ee_type;
        status = ucc_ee_executor_init(&params, &task->executor);
        if (UCC_OK != status) {
            ucc_error("failed to init executor: %s", ucc_status_string(status));
            goto coll_finalize;
        }
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
    ucc_assert(task->super.status == UCC_OPERATION_INITIALIZED);
    *request = &task->super;

    return UCC_OK;

coll_finalize:
    task->finalize(task);
    return status;
}

/* Check if user is trying to post the request which is either in completed,
   inprogress or error state.
   The only allowed case is: request is completed and has a
   persistent flag. Otherwise: bad usage. */
#define COLL_POST_STATUS_CHECK(_task) do {                              \
        if ((_task)->super.status != UCC_OPERATION_INITIALIZED) {       \
            if (!(UCC_OK == (_task)->super.status &&                    \
                  UCC_IS_PERSISTENT((_task)->bargs.args))) {            \
                ucc_error("%s request with invalid status %s is being posted", \
                          UCC_IS_PERSISTENT((_task)->bargs.args)        \
                          ? "persistent" : "non-persistent",            \
                          ucc_status_string((_task)->super.status));    \
                return UCC_ERR_INVALID_PARAM;                           \
            }                                                           \
        }                                                               \
    } while(0)

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_post, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);
    ucc_status_t status;
    ucc_debug("coll_post: req %p, seq_num %u", task, task->seq_num);

    COLL_POST_STATUS_CHECK(task);
    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        task->start_time = ucc_get_time();
    }

    if (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
        status = ucc_ee_executor_start(task->executor, NULL);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("failed to start executor: %s",
                      ucc_status_string(status));
        }
    }
    return task->post(task);
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_finalize, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);
    ucc_status_t st;

    ucc_debug("coll_finalize: req %p, seq_num %u", task, task->seq_num);
    if (ucc_unlikely(task->super.status == UCC_INPROGRESS)) {
        ucc_error("attempt to finalize task with status UCC_INPROGRESS");
        return UCC_ERR_INVALID_PARAM;
    }

    if (task->executor) {
        st = ucc_ee_executor_finalize(task->executor);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("executor finalize error: %s",
                      ucc_status_string(st));
        }
    }
    return task->finalize(task);
}

static ucc_status_t ucc_triggered_task_finalize(ucc_coll_task_t *task)
{
    ucc_trace("finalizing triggered ev task %p", task);
    ucc_free(task);
    return UCC_OK;
}

//NOLINTNEXTLINE
static void ucc_triggered_task_cb(void *task, ucc_status_t st)
{
    ucc_triggered_task_finalize((ucc_coll_task_t*)task);
}

static ucc_status_t ucc_triggered_coll_complete(ucc_coll_task_t *parent_task, //NOLINT
                                                ucc_coll_task_t *task)
{
    ucc_trace("triggered collective complete, task %p, seq_num %u",
              task, task->seq_num);
    if (!(task->flags & UCC_COLL_TASK_FLAG_EXECUTOR)) {
        /*  need to stop and finalize executor here in case if collective itself
         *  doesn't need executor and executor was created as part of
         *  triggered post
         */
        ucc_ee_executor_stop(task->executor);
        ucc_ee_executor_finalize(task->executor);
        task->executor = NULL;
    }
    return UCC_OK;
}

static ucc_status_t ucc_trigger_complete(ucc_coll_task_t *parent_task,
                                         ucc_coll_task_t *task)
{
    ucc_status_t status;

    ucc_trace("event triggered, ev_task %p, coll_task %p, seq_num %u",
              parent_task, task, task->seq_num);

    if (!(task->flags & UCC_COLL_TASK_FLAG_EXECUTOR)) {
        task->executor = parent_task->executor;
    }
    status = task->post(task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to post triggered coll, task %p, seq_num %u, %s",
                  task, task->seq_num, ucc_status_string(status));
        return status;
    }

    if (task->super.status == UCC_OK) {
        return ucc_triggered_coll_complete(task, task);
    } else {
        ucc_assert(task->super.status == UCC_INPROGRESS);
        // TODO use CB instead of EM
        ucc_event_manager_subscribe(&task->em, UCC_EVENT_COMPLETED, task,
                                    ucc_triggered_coll_complete);
    }
    return UCC_OK;
}

static void ucc_trigger_test(ucc_coll_task_t *task)
{
    ucc_status_t              status;
    ucc_ev_t                  post_event;
    ucc_ev_t                 *ev;
    ucc_ee_executor_params_t  params;

    if (task->ev == NULL) {
        if (task->ee->ee_type == UCC_EE_CUDA_STREAM || task->ee->ee_type == UCC_EE_ROCM_STREAM) {
            /* implicit event triggered */
            task->ev = (ucc_ev_t *) 0xFFFF; /* dummy event */
            task->executor = NULL;
        } else if (UCC_OK == ucc_ee_get_event_internal(task->ee, &ev,
                                                 &task->ee->event_in_queue)) {
            ucc_trace("triggered event arrived, ev_task %p", task);
            task->ev      = ev;
            task->ee_task = NULL;
        } else {
            task->status = UCC_OK;
            return;
        }
    }

    if (task->executor == NULL) {
        if (task->triggered_task->triggered_post_setup) {
            status = task->triggered_task->triggered_post_setup(task->triggered_task);
            if (ucc_unlikely(status != UCC_OK)) {
                ucc_error("error in triggered post setup, %s",
                          ucc_status_string(status));
                task->status = status;
                return;
            }
        }
        if (task->triggered_task->executor) {
            task->executor = task->triggered_task->executor;
        } else {
            /* triggered task doesn't need executor, init and start executor on
             * trigger task
             */
            params.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
            params.ee_type = task->ee->ee_type;
            status = ucc_ee_executor_init(&params, &task->executor);
            if (ucc_unlikely(status != UCC_OK)) {
                ucc_error("error in ee executor init, %s",
                           ucc_status_string(status));
                task->status = status;
                return;
            }
        }
        status = ucc_ee_executor_start(task->executor, task->ee->ee_context);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("error in ee executor start, %s",
                      ucc_status_string(status));
            task->status = status;
            return;
        }

        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.req             = &task->triggered_task->super;
        ucc_ee_set_event_internal(task->ee, &post_event,
                                  &task->ee->event_out_queue);
    }

    if (task->executor == NULL ||
        (ucc_ee_executor_status(task->executor) == UCC_OK)) {
        task->status = UCC_OK;
    }
}

ucc_status_t ucc_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                ucc_coll_task_t *task)
{
    ucc_coll_task_t *ev_task;

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
    ev_task->flags          = UCC_COLL_TASK_FLAG_CB;
    ev_task->cb.cb          = ucc_triggered_task_cb;
    ev_task->cb.data        = ev_task;
    ev_task->finalize       = ucc_triggered_task_finalize;
    ev_task->progress       = ucc_trigger_test;
    ev_task->status         = UCC_INPROGRESS;

    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        UCC_COLL_SET_TIMEOUT(ev_task, task->bargs.args.timeout);
    }
    ucc_event_manager_subscribe(&ev_task->em, UCC_EVENT_COMPLETED, task,
                                ucc_trigger_complete);

    return ucc_progress_queue_enqueue(UCC_TASK_CORE_CTX(ev_task)->pq, ev_task);
}

ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t *ev)
{
    ucc_coll_task_t *task = ucc_derived_of(ev->req, ucc_coll_task_t);

    ucc_debug("triggered_post: task %p, seq_num %u", task, task->seq_num);

    COLL_POST_STATUS_CHECK(task);
    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        task->start_time = ucc_get_time();
    }
    return task->triggered_post(ee, ev, task);
}
