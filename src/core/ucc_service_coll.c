/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_service_coll.h"
#include "ucc_team.h"
#include "ucc_global_opts.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_coll_utils.h"

uint64_t ucc_service_coll_map_cb(uint64_t ep, void *cb_ctx)
{
    ucc_service_coll_req_t *req  = cb_ctx;
    ucc_team_t             *team = req->team;
    ucc_rank_t              team_rank;

    team_rank = ucc_ep_map_eval(req->subset.map, (ucc_rank_t)ep);
    return ucc_ep_map_eval(team->ctx_map, team_rank);
}

static inline ucc_status_t
ucc_service_coll_req_init(ucc_team_t *team, ucc_subset_t *subset,
                          ucc_tl_team_t          **service_team,
                          ucc_service_coll_req_t **_req)
{
    ucc_context_t          *ctx = team->contexts[0];
    ucc_service_coll_req_t *req;

    *service_team = NULL;
    req = ucc_malloc(sizeof(*req), "service_req");
    if (!req) {
        ucc_error("failed to allocate %zd bytes for service coll req",
                  sizeof(*req));
        return UCC_ERR_NO_MEMORY;
    }
    req->team   = team;
    req->subset = *subset;

    if (ctx->service_team) {
        *service_team         = ctx->service_team;
        subset->map.type      = UCC_EP_MAP_CB;
        subset->map.cb.cb     = ucc_service_coll_map_cb;
        subset->map.cb.cb_ctx = req;
    } else {
        ucc_assert(team->service_team != NULL);
        *service_team = team->service_team;
    }

    *_req = req;
    return UCC_OK;
}

ucc_status_t ucc_service_allreduce(ucc_team_t *team, void *sbuf, void *rbuf,
                                   ucc_datatype_t dt, size_t count,
                                   ucc_reduction_op_t op, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req)
{
    ucc_tl_team_t  *steam;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t    status;

    status = ucc_service_coll_req_init(team, &subset, &steam, req);
    if (UCC_OK != status) {
        return status;
    }

    tl_iface = UCC_TL_TEAM_IFACE(steam);
    status = tl_iface->scoll.allreduce(&steam->super, sbuf, rbuf, dt, count, op,
                                       subset, &(*req)->task);
    if (status < 0) {
        ucc_free(*req);
        *req = NULL;
        ucc_error("failed to start service allreduce for team %p: %s", team,
                  ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_service_allgather(ucc_team_t *team, void *sbuf, void *rbuf,
                                   size_t msgsize, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req)
{
    ucc_tl_team_t  *steam;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t    status;

    status = ucc_service_coll_req_init(team, &subset, &steam, req);
    if (UCC_OK != status) {
        return status;
    }

    tl_iface = UCC_TL_TEAM_IFACE(steam);
    status   = tl_iface->scoll.allgather(&steam->super, sbuf, rbuf, msgsize,
                                       subset, &(*req)->task);
    if (status < 0) {
        ucc_free(*req);
        *req = NULL;
        ucc_error("failed to start service allgather for team %p: %s", team,
                  ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_service_bcast(ucc_team_t *team, void *buf, size_t msgsize,
                               ucc_rank_t root, ucc_subset_t subset,
                               ucc_service_coll_req_t **req)
{
    ucc_tl_team_t  *steam;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t    status;

    status = ucc_service_coll_req_init(team, &subset, &steam, req);
    if (UCC_OK != status) {
        return status;
    }

    tl_iface = UCC_TL_TEAM_IFACE(steam);
    status = tl_iface->scoll.bcast(&steam->super, buf, msgsize,
                                   root, subset, &(*req)->task);
    if (status < 0) {
        ucc_free(*req);
        *req = NULL;
        ucc_error("failed to start service bcast for team %p: %s", team,
                  ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_service_coll_test(ucc_service_coll_req_t *req)
{
    ucc_status_t status;

    status = ucc_collective_test(&req->task->super);
    if (UCC_INPROGRESS == status) {
        ucc_context_progress(req->team->contexts[0]);
    }
    return status;
}

ucc_status_t ucc_service_coll_finalize(ucc_service_coll_req_t *req)
{
    ucc_status_t status;

    status = ucc_collective_finalize_internal(req->task);
    ucc_free(req);
    return status;
}

typedef struct ucc_internal_oob_coll_info {
    ucc_team_t  *team;
    ucc_subset_t subset;
} ucc_internal_oob_coll_info_t;

static ucc_status_t ucc_internal_oob_allgather(void *sbuf, void *rbuf,
                                               size_t size, void *coll_info,
                                               void **request)
{
    ucc_internal_oob_coll_info_t *ci  = coll_info;
    ucc_service_coll_req_t       *req = NULL;
    ucc_status_t                  status;

    status =
        ucc_service_allgather(ci->team, sbuf, rbuf, size, ci->subset, &req);
    *request = (void *)req;
    return status;
}

static ucc_status_t ucc_internal_oob_test(void *request)
{
    ucc_service_coll_req_t *req = request;
    return ucc_service_coll_test(req);
}

static ucc_status_t ucc_internal_oob_free(void *request)
{
    ucc_service_coll_req_t *req = request;
    return ucc_service_coll_finalize(req);
}

ucc_status_t ucc_internal_oob_init(ucc_team_t *team, ucc_subset_t subset,
                                   ucc_team_oob_coll_t *oob)
{
    ucc_internal_oob_coll_info_t *ci;

    ci = ucc_malloc(sizeof(*ci), "internal_coll_info");
    if (!ci) {
        ucc_error("failed to allocate %zd bytes for internal_coll_info",
                  sizeof(*ci));
        return UCC_ERR_NO_MEMORY;
    }

    ci->team       = team;
    ci->subset     = subset;
    oob->coll_info = ci;
    oob->allgather = ucc_internal_oob_allgather;
    oob->req_test  = ucc_internal_oob_test;
    oob->req_free  = ucc_internal_oob_free;
    oob->n_oob_eps = (uint32_t)subset.map.ep_num;
    oob->oob_ep    = (uint32_t)subset.myrank;

    return UCC_OK;
}

void ucc_internal_oob_finalize(ucc_team_oob_coll_t *oob)
{
    ucc_free(oob->coll_info);
}

/* Helper macro to get dt_check from schedule */
#define UCC_DT_CHECK_SCHEDULE(_task) \
    ucc_derived_of((_task)->schedule, ucc_dt_check_schedule_t)

#define UCC_DT_CHECK_FROM_TASK(_task) \
    (&UCC_DT_CHECK_SCHEDULE(_task)->dt_check)

static ucc_status_t ucc_dt_validate_results(ucc_dt_check_state_t *dt_check)
{
    int16_t *values;

    if (!dt_check) {
        return UCC_ERR_INVALID_PARAM;
    }
    values = dt_check->values;
    if (values[0] == (int16_t) UCC_ERR_NOT_SUPPORTED) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (values[0] != -values[1]) {
        return UCC_ERR_INVALID_PARAM;
    }
    if (values[2] != -values[3]) {
        return UCC_ERR_INVALID_PARAM;
    }
    return UCC_OK;
}

static ucc_status_t ucc_dt_check_allreduce_post(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = UCC_DT_CHECK_FROM_TASK(allreduce_wrapper);
    ucc_team_t           *team     = allreduce_wrapper->bargs.team;
    ucc_status_t          status;

    if (!dt_check) {
        allreduce_wrapper->status = UCC_ERR_INVALID_PARAM;
        return UCC_ERR_INVALID_PARAM;
    }

    status = ucc_service_allreduce(team, dt_check->values, dt_check->values,
                                   UCC_DT_INT16, 4, UCC_OP_MIN,
                                   dt_check->subset, &dt_check->check_req);
    if (status != UCC_OK) {
        ucc_schedule_t *schedule = allreduce_wrapper->schedule;

        allreduce_wrapper->status = status;
        ucc_task_complete(allreduce_wrapper);
        /* ucc_schedule_start already set the schedule to UCC_INPROGRESS before
         * firing SCHEDULE_STARTED.  Fail the whole schedule now so that
         * ucc_collective_finalize_internal will not refuse to run because the
         * top-level request is still UCC_INPROGRESS.
         * Note: ucc_task_complete above does not emit UCC_EVENT_COMPLETED_SCHEDULE
         * on error (status < 0), so the schedule counter is not incremented and
         * the schedule must be force-completed here. */
        if (schedule) {
            schedule->n_completed_tasks = schedule->n_tasks;
            schedule->super.status      = status;
            ucc_task_complete(&schedule->super);
        }
        return status;
    }
    allreduce_wrapper->status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(team->contexts[0]->pq, allreduce_wrapper);
}

static void ucc_dt_check_allreduce_progress(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = UCC_DT_CHECK_FROM_TASK(allreduce_wrapper);
    ucc_coll_task_t      *ar_task;
    ucc_status_t          status;

    if (!dt_check || !dt_check->check_req) {
        allreduce_wrapper->status = UCC_ERR_INVALID_PARAM;
        return;
    }

    ar_task = dt_check->check_req->task;
    status  = ar_task->super.status;
    if (status == UCC_INPROGRESS) {
        allreduce_wrapper->status = UCC_INPROGRESS;
        return;
    }

    ucc_service_coll_finalize(dt_check->check_req);
    dt_check->check_req = NULL;

    if (status != UCC_OK) {
        dt_check->ar_status       = status;
        dt_check->validated       = 0;
        allreduce_wrapper->status = UCC_OK;
        return;
    }

    status = ucc_dt_validate_results(dt_check);
    dt_check->validated       = (status == UCC_OK);
    allreduce_wrapper->status = UCC_OK;
}

static ucc_status_t ucc_dt_check_allreduce_finalize(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = UCC_DT_CHECK_FROM_TASK(allreduce_wrapper);

    if (dt_check && dt_check->check_req) {
        ucc_service_coll_finalize(dt_check->check_req);
        dt_check->check_req = NULL;
    }
    ucc_mpool_put(allreduce_wrapper);
    return UCC_OK;
}

static ucc_status_t ucc_dt_check_actual_wrapper_post(ucc_coll_task_t *wrapper)
{
    ucc_dt_check_state_t *dt_check    = UCC_DT_CHECK_FROM_TASK(wrapper);
    ucc_coll_task_t      *actual_task = dt_check->actual_task;
    ucc_schedule_t       *schedule    = wrapper->schedule;
    ucc_status_t          status;

    if (!dt_check->validated) {
        ucc_status_t err;
        /* Use the real error when init failed for non-DT reasons, or when
         * the service allreduce itself failed.  Otherwise it is a genuine
         * datatype-consistency mismatch across ranks. */
        if (dt_check->init_status != UCC_OK) {
            err = dt_check->init_status;
        } else if (dt_check->ar_status != UCC_OK) {
            err = dt_check->ar_status;
        } else {
            err = UCC_ERR_NOT_SUPPORTED;
        }
        wrapper->status = err;
        ucc_task_complete(wrapper);
        if (schedule) {
            /* Prevent ucc_schedule_completed_handler from calling
             * ucc_task_complete a second time after allreduce_wrapper
             * fires UCC_EVENT_COMPLETED_SCHEDULE. */
            schedule->n_completed_tasks = schedule->n_tasks;
            schedule->super.status      = err;
            ucc_task_complete(&schedule->super);
        }
        return UCC_OK;
    }

    /* validated=1 implies local init succeeded, so actual_task must be set */
    ucc_assert(actual_task != NULL);

    /* Transfer executor from schedule to actual_task before posting */
    if (actual_task->flags & UCC_COLL_TASK_FLAG_EXECUTOR) {
        ucc_coll_task_t *sched_task = &schedule->super;
        actual_task->executor       = sched_task->executor;
        actual_task->flags         |= UCC_COLL_TASK_FLAG_EXECUTOR_STOP;
        sched_task->flags          &= ~((uint32_t)UCC_COLL_TASK_FLAG_EXECUTOR_STOP);
    }

    status = actual_task->post(actual_task);
    if (status < 0) {
        wrapper->status = status;
        if (schedule) {
            /* EXECUTOR_STOP was transferred from the schedule to actual_task
             * above.  Since actual_task never ran, the executor was started
             * (in ucc_collective_post) but is still owned by the schedule.
             * Return the stop responsibility to the schedule so that
             * ucc_task_complete can stop it. */
            if (actual_task->flags & UCC_COLL_TASK_FLAG_EXECUTOR_STOP) {
                ucc_coll_task_t *sched_task = &schedule->super;

                sched_task->executor = actual_task->executor;
                sched_task->flags   |= UCC_COLL_TASK_FLAG_EXECUTOR_STOP;
                actual_task->flags  &= ~(uint32_t)UCC_COLL_TASK_FLAG_EXECUTOR_STOP;
            }
            ucc_task_complete(wrapper);
            schedule->n_completed_tasks = schedule->n_tasks;
            schedule->super.status      = status;
            ucc_task_complete(&schedule->super);
        }
        return status;
    }
    wrapper->status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(wrapper->bargs.team->contexts[0]->pq, wrapper);
}

static void ucc_dt_check_actual_wrapper_progress(ucc_coll_task_t *wrapper)
{
    ucc_dt_check_state_t *dt_check    = UCC_DT_CHECK_FROM_TASK(wrapper);
    ucc_coll_task_t      *actual_task = dt_check->actual_task;

    if (ucc_unlikely(!actual_task)) {
        wrapper->status = UCC_ERR_NOT_SUPPORTED;
        return;
    }
    wrapper->status = actual_task->status;
}

static ucc_status_t ucc_dt_check_actual_wrapper_finalize(ucc_coll_task_t *wrapper)
{
    ucc_dt_check_state_t *dt_check    = UCC_DT_CHECK_FROM_TASK(wrapper);
    ucc_coll_task_t      *actual_task = dt_check->actual_task;
    ucc_status_t          status      = UCC_OK;

    if (actual_task && actual_task->finalize) {
        status = actual_task->finalize(actual_task);
    }
    ucc_mpool_put(wrapper);
    return status;
}

static ucc_status_t ucc_dt_check_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_dt_check_schedule_t *dt_schedule =
        ucc_derived_of(task, ucc_dt_check_schedule_t);
    ucc_status_t             status;

    status = ucc_schedule_finalize(task);
    ucc_coll_task_destruct(&dt_schedule->super.super);
    ucc_free(dt_schedule);
    return status;
}

ucc_coll_task_t* ucc_service_dt_check(ucc_team_t            *team,
                                      const ucc_coll_args_t *coll_args,
                                      ucc_status_t           local_status,
                                      ucc_coll_task_t       *task,
                                      ucc_status_t          *status_out)
{
    ucc_rank_t               rank           = team->rank;
    ucc_rank_t               root           = coll_args->root;
    ucc_coll_type_t          coll_type      = coll_args->coll_type;
    ucc_datatype_t           local_dt       = UCC_DT_INT8;
    ucc_memory_type_t        local_mem_type = UCC_MEMORY_TYPE_UNKNOWN;
    ucc_dt_check_schedule_t *dt_schedule;
    ucc_dt_check_state_t    *dt_check;
    ucc_coll_task_t         *allreduce_wrapper;
    ucc_coll_task_t         *actual_wrapper;
    ucc_base_coll_args_t     schedule_bargs;
    ucc_base_coll_args_t     empty_bargs;
    ucc_base_team_t         *base_team;
    ucc_status_t             status;

    /* Read the local signature from coll_args (valid even when task is NULL) */
    if (rank == root) {
        switch (coll_type) {
        case UCC_COLL_TYPE_GATHER:
            if (UCC_IS_INPLACE(*coll_args)) {
                local_dt       = coll_args->dst.info.datatype;
                local_mem_type = coll_args->dst.info.mem_type;
            } else {
                local_dt       = coll_args->src.info.datatype;
                local_mem_type = coll_args->src.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_GATHERV:
            if (UCC_IS_INPLACE(*coll_args)) {
                local_dt       = coll_args->dst.info_v.datatype;
                local_mem_type = coll_args->dst.info_v.mem_type;
            } else {
                local_dt       = coll_args->src.info.datatype;
                local_mem_type = coll_args->src.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_SCATTER:
            if (UCC_IS_INPLACE(*coll_args)) {
                local_dt       = coll_args->src.info.datatype;
                local_mem_type = coll_args->src.info.mem_type;
            } else {
                local_dt       = coll_args->dst.info.datatype;
                local_mem_type = coll_args->dst.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_SCATTERV:
            if (UCC_IS_INPLACE(*coll_args)) {
                local_dt       = coll_args->src.info_v.datatype;
                local_mem_type = coll_args->src.info_v.mem_type;
            } else {
                local_dt       = coll_args->dst.info.datatype;
                local_mem_type = coll_args->dst.info.mem_type;
            }
            break;
        default:
            break;
        }
    } else {
        switch (coll_type) {
        case UCC_COLL_TYPE_GATHER:
        case UCC_COLL_TYPE_GATHERV:
            local_dt       = coll_args->src.info.datatype;
            local_mem_type = coll_args->src.info.mem_type;
            break;
        case UCC_COLL_TYPE_SCATTER:
        case UCC_COLL_TYPE_SCATTERV:
            local_dt       = coll_args->dst.info.datatype;
            local_mem_type = coll_args->dst.info.mem_type;
            break;
        default:
            break;
        }
    }

    /* Build bargs for the schedule.  When task is NULL (local init failed),
     * construct minimal bargs from coll_args; scratch was freed by the caller
     * before this call.  When task is valid, reuse its bargs so the schedule
     * inherits the scratch pointer and other fields set during init. */
    if (task != NULL) {
        schedule_bargs = task->bargs;
        base_team      = task->team;
    } else {
        memset(&schedule_bargs, 0, sizeof(schedule_bargs));
        schedule_bargs.team = team;
        memcpy(&schedule_bargs.args, coll_args, sizeof(*coll_args));
        base_team = NULL;
    }

    dt_schedule = (ucc_dt_check_schedule_t *)ucc_malloc(sizeof(*dt_schedule),
                                                         "dt_check_schedule");
    if (!dt_schedule) {
        ucc_error("failed to allocate dt_check_schedule");
        if (status_out) {
            *status_out = UCC_ERR_NO_MEMORY;
        }
        return NULL;
    }
    memset(dt_schedule, 0, sizeof(*dt_schedule));

    ucc_coll_task_construct(&dt_schedule->super.super);
    if (base_team != NULL) {
        status = ucc_schedule_init(&dt_schedule->super, &schedule_bargs,
                                   base_team);
    } else {
        /* base_team is NULL (local init failed): manually init the schedule
         * to avoid dereferencing NULL in ucc_schedule_init → team->context. */
        status = ucc_coll_task_init(&dt_schedule->super.super,
                                    &schedule_bargs, NULL);
        if (status == UCC_OK) {
            dt_schedule->super.super.flags |= UCC_COLL_TASK_FLAG_IS_SCHEDULE;
            dt_schedule->super.ctx          = team->contexts[0];
            dt_schedule->super.n_tasks      = 0;
            dt_schedule->super.n_completed_tasks = 0;
        }
    }
    if (status != UCC_OK) {
        ucc_error("failed to initialize dt_check schedule: %s",
                  ucc_status_string(status));
        ucc_free(dt_schedule);
        if (status_out) {
            *status_out = status;
        }
        return NULL;
    }

    dt_check = &dt_schedule->dt_check;
    /* Use the failure sentinel when local init failed OR the DT is
     * non-predefined.  A rank that failed init must still participate in
     * the allreduce so that all ranks get a uniform result and none hang.
     * Save the real error for non-DT init failures (OOM, invalid args,
     * etc.) so the failure path can propagate the accurate status. */
    if (local_status != UCC_OK || !UCC_DT_IS_PREDEFINED(local_dt)) {
        dt_check->values[0]  = (int16_t) UCC_ERR_NOT_SUPPORTED;
        dt_check->values[1]  = -(int16_t) UCC_ERR_NOT_SUPPORTED;
        /* Record any hard init error (not UCC_ERR_NOT_SUPPORTED, which is the
         * normal DT-mismatch sentinel) so actual_wrapper_post can surface the
         * right status code.  This is independent of whether the DT is
         * predefined: a rank that fails with OOM on a non-predefined DT must
         * still report OOM, not UCC_ERR_NOT_SUPPORTED. */
        dt_check->init_status = (local_status != UCC_OK &&
                                  local_status != UCC_ERR_NOT_SUPPORTED)
                                 ? local_status : UCC_OK;
    } else {
        dt_check->values[0]   = (int16_t) local_dt;
        dt_check->values[1]   = -(int16_t) local_dt;
        dt_check->init_status = UCC_OK;
    }
    dt_check->values[2]         = (int16_t) local_mem_type;
    dt_check->values[3]         = -(int16_t) local_mem_type;
    dt_check->subset.myrank     = team->rank;
    dt_check->subset.map.type   = UCC_EP_MAP_FULL;
    dt_check->subset.map.ep_num = team->size;
    dt_check->check_req         = NULL;
    dt_check->validated         = 0;
    dt_check->ar_status         = UCC_OK;
    dt_check->actual_task       = task;   /* may be NULL when local init failed */

    memset(&empty_bargs, 0, sizeof(empty_bargs));
    empty_bargs.team = team;

    allreduce_wrapper = ucc_mpool_get(&team->contexts[0]->lib->stub_tasks_mp);
    if (!allreduce_wrapper) {
        ucc_error("failed to allocate allreduce wrapper task from mpool");
        status = UCC_ERR_NO_MEMORY;
        goto error_schedule;
    }
    status = ucc_coll_task_init(allreduce_wrapper, &empty_bargs, base_team);
    if (status != UCC_OK) {
        ucc_error("failed to init allreduce wrapper task: %s",
                  ucc_status_string(status));
        ucc_mpool_put(allreduce_wrapper);
        goto error_schedule;
    }
    allreduce_wrapper->post     = ucc_dt_check_allreduce_post;
    allreduce_wrapper->progress = ucc_dt_check_allreduce_progress;
    allreduce_wrapper->finalize = ucc_dt_check_allreduce_finalize;
    allreduce_wrapper->status   = UCC_OPERATION_INITIALIZED;
    status = ucc_task_subscribe_dep(&dt_schedule->super.super, allreduce_wrapper,
                                    UCC_EVENT_SCHEDULE_STARTED);
    if (status != UCC_OK) {
        ucc_error("failed to subscribe allreduce wrapper: %s",
                  ucc_status_string(status));
        goto error_allreduce_wrapper;
    }
    status = ucc_schedule_add_task(&dt_schedule->super, allreduce_wrapper);
    if (status != UCC_OK) {
        ucc_error("failed to add allreduce wrapper to schedule: %s",
                  ucc_status_string(status));
        goto error_allreduce_wrapper;
    }

    actual_wrapper = ucc_mpool_get(&team->contexts[0]->lib->stub_tasks_mp);
    if (!actual_wrapper) {
        ucc_error("failed to allocate actual wrapper task from mpool");
        status = UCC_ERR_NO_MEMORY;
        goto error_allreduce_wrapper;
    }
    status = ucc_coll_task_init(actual_wrapper, &empty_bargs, base_team);
    if (status != UCC_OK) {
        ucc_error("failed to init actual wrapper task: %s",
                  ucc_status_string(status));
        ucc_mpool_put(actual_wrapper);
        goto error_allreduce_wrapper;
    }
    actual_wrapper->post     = ucc_dt_check_actual_wrapper_post;
    actual_wrapper->progress = ucc_dt_check_actual_wrapper_progress;
    actual_wrapper->finalize = ucc_dt_check_actual_wrapper_finalize;
    actual_wrapper->status   = UCC_OPERATION_INITIALIZED;
    status = ucc_task_subscribe_dep(allreduce_wrapper, actual_wrapper,
                                    UCC_EVENT_COMPLETED);
    if (status != UCC_OK) {
        ucc_error("failed to subscribe actual wrapper dependency: %s",
                  ucc_status_string(status));
        ucc_mpool_put(actual_wrapper);
        goto error_allreduce_wrapper;
    }
    status = ucc_schedule_add_task(&dt_schedule->super, actual_wrapper);
    if (status != UCC_OK) {
        ucc_error("failed to add actual wrapper to schedule: %s",
                  ucc_status_string(status));
        ucc_mpool_put(actual_wrapper);
        goto error_allreduce_wrapper;
    }

    if (task != NULL && (task->flags & UCC_COLL_TASK_FLAG_EXECUTOR)) {
        dt_schedule->super.super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    }

    dt_schedule->super.super.post     = ucc_schedule_start;
    dt_schedule->super.super.progress = NULL;
    dt_schedule->super.super.finalize = ucc_dt_check_schedule_finalize;
    return &dt_schedule->super.super;

error_allreduce_wrapper:
    ucc_coll_task_destruct(allreduce_wrapper);
    ucc_mpool_put(allreduce_wrapper);
error_schedule:
    ucc_coll_task_destruct(&dt_schedule->super.super);
    ucc_free(dt_schedule);
    if (status_out) {
        *status_out = status;
    }
    return NULL;
}
