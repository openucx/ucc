/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
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
    case UCC_COLL_TYPE_BCAST:
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        if (UCC_IS_INPLACE(*coll_args)) {
            ucc_warn("Inplace flag for %s is not defined by UCC API",
                     ucc_coll_type_str(coll_args->coll_type));
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

#define UCC_COLL_TYPE_SKIP_ZERO_SIZE \
    (UCC_COLL_TYPE_ALLREDUCE |       \
     UCC_COLL_TYPE_ALLGATHER |       \
     UCC_COLL_TYPE_ALLTOALL |        \
     UCC_COLL_TYPE_BCAST |           \
     UCC_COLL_TYPE_GATHER |          \
     UCC_COLL_TYPE_REDUCE |          \
     UCC_COLL_TYPE_SCATTER)

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_init,
                      (coll_args, request, team), ucc_coll_args_t *coll_args,
                      ucc_coll_req_h *request, ucc_team_h team)
{
    ucc_base_coll_args_t      op_args = {0};
    ucc_coll_task_t          *task;
    ucc_status_t              status;
    ucc_ee_executor_params_t  params;
    ucc_memory_type_t         coll_mem_type;
    ucc_ee_type_t             coll_ee_type;
    size_t                    coll_size;

    if (ucc_unlikely(team->state != UCC_TEAM_ACTIVE)) {
        ucc_error("team %p is used before team create is completed", team);
        return UCC_ERR_INVALID_PARAM;
    }
    /* Global check to reduce the amount of checks throughout
       all TLs */

    if (UCC_COLL_TYPE_SKIP_ZERO_SIZE & coll_args->coll_type) {
        coll_size = ucc_coll_args_msgsize(coll_args, team->rank, team->size);
        if (coll_size == 0) {
            task = ucc_mpool_get(&team->contexts[0]->lib->stub_tasks_mp);
            if (ucc_unlikely(!task)) {
                ucc_error("failed to allocate dummy task");
                return UCC_ERR_NO_MEMORY;
            }
            op_args.mask = 0;
            memcpy(&op_args.args, coll_args, sizeof(ucc_coll_args_t));
            op_args.team = team;
            op_args.args.flags = 0;
            UCC_COPY_PARAM_BY_FIELD(&op_args.args, coll_args,
                                    UCC_COLL_ARGS_FIELD_FLAGS, flags);
            ucc_coll_task_init(task, &op_args, NULL);
            goto print_trace;
        }
    }

    if (UCC_COLL_ARGS_ACTIVE_SET(coll_args) &&
        (UCC_COLL_TYPE_BCAST != coll_args->coll_type)) {
        ucc_warn("Active Sets are only supported for bcast");
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
    memcpy(&op_args.args, coll_args, sizeof(ucc_coll_args_t));
    op_args.team = team;

    op_args.args.flags = 0;
    UCC_COPY_PARAM_BY_FIELD(&op_args.args, coll_args, UCC_COLL_ARGS_FIELD_FLAGS,
                            flags);

    if (!ucc_coll_args_is_mem_symmetric(&op_args.args, team->rank) &&
        ucc_coll_args_is_rooted(op_args.args.coll_type)) {
        status = ucc_coll_args_init_asymmetric_buffer(&op_args.args, team,
                    &op_args.asymmetric_save_info);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("handling asymmetric memory failed");
            return status;
        }
    } else {
        op_args.asymmetric_save_info.scratch = NULL;
    }

    status = ucc_coll_init(team->score_map, &op_args, &task);
    if (UCC_ERR_NOT_SUPPORTED == status) {
        ucc_debug("failed to init collective: not supported");
        goto free_scratch;
    } else if (ucc_unlikely(status < 0)) {
        char coll_args_str[256] = {0};
        ucc_coll_args_str(&op_args.args, team->rank, team->size, coll_args_str,
                          sizeof(coll_args_str));
        ucc_error("failed to init collective: %s, err: (%d) %s", coll_args_str,
                  status, ucc_status_string(status));
        goto free_scratch;
    }

    task->flags |= UCC_COLL_TASK_FLAG_TOP_LEVEL;
    if (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
        task->flags |= UCC_COLL_TASK_FLAG_EXECUTOR_STOP;
        coll_mem_type = ucc_coll_args_mem_type(&op_args.args, team->rank);
        switch(coll_mem_type) {
        case UCC_MEMORY_TYPE_CUDA:
        case UCC_MEMORY_TYPE_CUDA_MANAGED:
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

    ucc_assert(task->super.status == UCC_OPERATION_INITIALIZED);

print_trace:
    *request = &task->super;
    if (ucc_unlikely(ucc_global_config.coll_trace.log_level >=
                     UCC_LOG_LEVEL_DIAG)) {
        char coll_str[256];
        ucc_coll_str(task, coll_str, sizeof(coll_str),
                     ucc_global_config.coll_trace.log_level);
        if (ucc_global_config.coll_trace.log_level <= UCC_LOG_LEVEL_INFO) {
            if (team->rank == 0) {
                ucc_log_component_collective_trace(
                    ucc_global_config.coll_trace.log_level, "coll_init: %s",
                    coll_str);
            }
        } else {
            ucc_coll_trace_debug("coll_init: %s", coll_str);
        }
    }

    return UCC_OK;

coll_finalize:
    task->finalize(task);
free_scratch:
    if (op_args.asymmetric_save_info.scratch != NULL) {
        ucc_mc_free(op_args.asymmetric_save_info.scratch);
    }
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

    if (ucc_global_config.coll_trace.log_level >= UCC_LOG_LEVEL_DEBUG) {
        ucc_rank_t rank = task->bargs.team->rank;
        if (ucc_global_config.coll_trace.log_level == UCC_LOG_LEVEL_DEBUG) {
            if (rank == 0) {
                ucc_log_component_collective_trace(
                    ucc_global_config.coll_trace.log_level,
                    "coll post: req %p, seq_num %u", task, task->seq_num);
            }
        } else {
            ucc_log_component_collective_trace(
                ucc_global_config.coll_trace.log_level,
                "coll post: rank %d req %p, seq_num %u", rank, task,
                task->seq_num);
        }
    }

    if (task->bargs.asymmetric_save_info.scratch != NULL &&
        (task->bargs.args.coll_type == UCC_COLL_TYPE_SCATTER ||
         task->bargs.args.coll_type == UCC_COLL_TYPE_SCATTERV)) {
        status = ucc_copy_asymmetric_buffer(task);
        if (status != UCC_OK) {
            ucc_error("failure copying in asymmetric buffer: %s",
                        ucc_status_string(status));
            return status;
        }
    }

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

ucc_status_t ucc_collective_triggered_post(ucc_ee_h ee, ucc_ev_t *ev)
{
    ucc_coll_task_t *task = ucc_derived_of(ev->req, ucc_coll_task_t);

    if (ucc_global_config.coll_trace.log_level >= UCC_LOG_LEVEL_DEBUG) {
        ucc_rank_t rank = task->bargs.team->rank;
        if (ucc_global_config.coll_trace.log_level == UCC_LOG_LEVEL_DEBUG) {
            if (rank == 0) {
                ucc_log_component_collective_trace(
                    ucc_global_config.coll_trace.log_level,
                    "coll triggered_post: req %p, seq_num %u", task,
                    task->seq_num);
            }
        } else {
            ucc_log_component_collective_trace(
                ucc_global_config.coll_trace.log_level,
                "coll triggered_post: rank %d req %p, seq_num %u", rank, task,
                task->seq_num);
        }
    }

    COLL_POST_STATUS_CHECK(task);
    if (UCC_COLL_TIMEOUT_REQUIRED(task)) {
        task->start_time = ucc_get_time();
    }
    return task->triggered_post(ee, ev, task);
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_init_and_post,
                      (coll_args, request, team), ucc_coll_args_t *coll_args, //NOLINT
                      ucc_coll_req_h *request, ucc_team_h team) //NOLINT
{
    ucc_error("ucc_collective_init_and_post() is not implemented");

    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_collective_finalize_internal(ucc_coll_task_t *task)
{
    ucc_status_t st;

    if (ucc_unlikely(task->super.status == UCC_INPROGRESS)) {
        ucc_error("attempt to finalize task with status UCC_INPROGRESS");
        return UCC_ERR_INVALID_PARAM;
    }

    if (task->bargs.asymmetric_save_info.scratch) {
        st = ucc_coll_args_free_asymmetric_buffer(task);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("error freeing asymmetric buf: %s", ucc_status_string(st));
        }
    }

    if (task->executor) {
        st = ucc_ee_executor_finalize(task->executor);
        if (ucc_unlikely(st != UCC_OK)) {
            ucc_error("executor finalize error: %s", ucc_status_string(st));
        }
    }
    return task->finalize(task);
}

UCC_CORE_PROFILE_FUNC(ucc_status_t, ucc_collective_finalize, (request),
                      ucc_coll_req_h request)
{
    ucc_coll_task_t *task = ucc_derived_of(request, ucc_coll_task_t);

    if (ucc_global_config.coll_trace.log_level >= UCC_LOG_LEVEL_DEBUG) {
        if (task->team) {
            ucc_rank_t rank = task->team->params.team->rank;
            if (ucc_global_config.coll_trace.log_level == UCC_LOG_LEVEL_DEBUG) {
                if (rank == 0) {
                    ucc_log_component_collective_trace(
                        ucc_global_config.coll_trace.log_level,
                        "coll finalize: req %p, seq_num %u", task, task->seq_num);
                }
            } else {
                ucc_log_component_collective_trace(
                    ucc_global_config.coll_trace.log_level,
                    "coll finalize: rank %d req %p, seq_num %u", rank, task,
                    task->seq_num);
            }
        }
    }
    return ucc_collective_finalize_internal(task);
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

static ucc_status_t ucc_trigger_complete(ucc_coll_task_t *parent_task,
                                         ucc_coll_task_t *task)
{
    ucc_status_t status;

    ucc_trace("event triggered, ev_task %p, coll_task %p, seq_num %u",
              parent_task, task, task->seq_num);

    if (!(task->flags & UCC_COLL_TASK_FLAG_EXECUTOR)) {
        task->executor = parent_task->executor;
        task->flags |= (UCC_COLL_TASK_FLAG_EXECUTOR_STOP |
                        UCC_COLL_TASK_FLAG_EXECUTOR_DESTROY);
    }

    status = task->post(task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_error("failed to post triggered coll, task %p, seq_num %u, %s",
                  task, task->seq_num, ucc_status_string(status));
    }
    return status;
}

static void ucc_trigger_test(ucc_coll_task_t *task)
{
    ucc_status_t              status;
    ucc_ev_t                  post_event;
    ucc_ev_t                 *ev;
    ucc_ee_executor_params_t  params;

    if (task->ev == NULL) {
        if ((task->ee->ee_type == UCC_EE_CUDA_STREAM) ||
            (task->ee->ee_type == UCC_EE_ROCM_STREAM)) {
            /* implicit event triggered */
            task->ev       = (ucc_ev_t *) 0xFFFF; /* dummy event */
            task->executor = NULL;
        } else if (UCC_OK == ucc_ee_get_event_internal(task->ee, &ev,
                                                 &task->ee->event_in_queue)) {
            ucc_trace("triggered event arrived, ev_task %p", task);
            task->ev       = ev;
            task->executor = NULL;
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
            params.mask       = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE |
                                UCC_EE_EXECUTOR_PARAM_FIELD_TASK_TYPES;
            params.ee_type    = task->ee->ee_type;
            params.task_types = 0;
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
        post_event.ev_context      = NULL;
        post_event.req             = &task->triggered_task->super;
        ucc_ee_set_event_internal(task->ee, &post_event,
                                  &task->ee->event_out_queue);
    }

    if (ucc_ee_executor_status(task->executor) == UCC_OK) {
        task->status = UCC_OK;
    }
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
    task->ee           = ee;
    task->super.status = UCC_OPERATION_INITIALIZED;
    ev_task = ucc_malloc(sizeof(*ev_task), "ev_task");
    if (!ev_task) {
        ucc_error("failed to allocate %zd bytes for ev_task",
                  sizeof(*ev_task));
        return UCC_ERR_NO_MEMORY;
    }

    ucc_coll_task_construct(ev_task);
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
    status = ucc_event_manager_subscribe(ev_task, UCC_EVENT_COMPLETED, task,
                                         ucc_trigger_complete);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    return ucc_progress_queue_enqueue(task->bargs.team->contexts[0]->pq, ev_task);
}
