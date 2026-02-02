/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_service_coll.h"
#include "ucc_team.h"
#include "ucc_global_opts.h"
#include "schedule/ucc_schedule.h"
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
        ucc_error("failed to start service allreduce for team %p: %s", team,
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

/**
 * Validate allreduced datatype values using min/max trick
 *
 * After MIN allreduce on [dt, -dt, mem, -mem]:
 *   - values[0] contains min(dt) across all ranks
 *   - values[1] contains min(-dt) = -max(dt) across all ranks
 *   - If values[0] == -values[1], all ranks have identical dt
 *   - Same logic applies to memory type with values[2] and values[3]
 */
static ucc_status_t ucc_dt_validate_results(ucc_dt_check_state_t *dt_check)
{
    int16_t *values;

    /* Safety checks */
    if (!dt_check) {
        return UCC_ERR_INVALID_PARAM;
    }
    values = dt_check->values;
    /* Check if any rank has non-contiguous datatype */
    if (values[0] == (int16_t) UCC_ERR_NOT_SUPPORTED) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    /* Check if all ranks have the same datatype using min/max trick */
    if (values[0] != -values[1]) {
        return UCC_ERR_INVALID_PARAM;
    }
    /* Check if all ranks have the same memory type */
    if (values[2] != -values[3]) {
        return UCC_ERR_INVALID_PARAM;
    }
    return UCC_OK;
}

/**
 * Post function for allreduce wrapper task using service allreduce
 *
 * Starts the service allreduce (MIN) to detect datatype mismatches across all ranks.
 */
static ucc_status_t ucc_dt_check_allreduce_post(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = UCC_DT_CHECK_FROM_TASK(allreduce_wrapper);
    ucc_team_t           *team     = allreduce_wrapper->bargs.team;
    ucc_status_t          status;

    /* Safety check */
    if (!dt_check) {
        allreduce_wrapper->status = UCC_ERR_INVALID_PARAM;
        return UCC_ERR_INVALID_PARAM;
    }

    /* Start in-place service allreduce with MIN operation on 4 int16_t values */
    status = ucc_service_allreduce(team, dt_check->values, dt_check->values,
                                   UCC_DT_INT16, 4, UCC_OP_MIN,
                                   dt_check->subset, &dt_check->check_req);
    if (status != UCC_OK) {
        allreduce_wrapper->status = status;
        return status;
    }
    allreduce_wrapper->status = UCC_INPROGRESS;
    /* Enqueue wrapper task for progress */
    return ucc_progress_queue_enqueue(team->contexts[0]->pq, allreduce_wrapper);
}

/**
 * Progress function for allreduce wrapper task using service allreduce
 *
 * Progresses service allreduce, and when complete, validates the reduced datatypes.
 */
static void ucc_dt_check_allreduce_progress(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = UCC_DT_CHECK_FROM_TASK(allreduce_wrapper);
    ucc_coll_task_t      *ar_task;
    ucc_status_t          status;

    /* Safety check */
    if (!dt_check || !dt_check->check_req) {
        allreduce_wrapper->status = UCC_ERR_INVALID_PARAM;
        return;
    }

    /* Check status of the service allreduce */
    ar_task = dt_check->check_req->task;
    status  = ar_task->super.status;
    if (status == UCC_INPROGRESS) {
        allreduce_wrapper->status = UCC_INPROGRESS;
        return;
    }

    /* Service allreduce completed (or failed) - finalize it */
    ucc_service_coll_finalize(dt_check->check_req);
    dt_check->check_req = NULL;

    /* If service allreduce failed, mark validation as failed */
    if (status != UCC_OK) {
        dt_check->validated = 0;
        allreduce_wrapper->status = status;
        return;
    }

    /* Service allreduce succeeded - validate using min/max check */
    status = ucc_dt_validate_results(dt_check);
    dt_check->validated = (status == UCC_OK);
    /* Completes with UCC_OK so schedule continues to actual wrapper */
    allreduce_wrapper->status = UCC_OK;
}

/**
 * Finalize function for allreduce wrapper task
 */
static ucc_status_t ucc_dt_check_allreduce_finalize(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = UCC_DT_CHECK_FROM_TASK(allreduce_wrapper);

    /* Clean up check_req if it wasn't finalized in progress */
    if (dt_check && dt_check->check_req) {
        ucc_service_coll_finalize(dt_check->check_req);
        dt_check->check_req = NULL;
    }
    /* dt_check is embedded in schedule, no need to free */
    /* Return wrapper task to memory pool */
    ucc_mpool_put(allreduce_wrapper);
    return UCC_OK;
}

/**
 * Post function for actual task wrapper
 *
 * Checks validation result and only posts actual task if validation succeeded.
 */
static ucc_status_t ucc_dt_check_actual_wrapper_post(ucc_coll_task_t *wrapper)
{
    ucc_dt_check_state_t *dt_check    = UCC_DT_CHECK_FROM_TASK(wrapper);
    ucc_coll_task_t      *actual_task = dt_check->actual_task;
    ucc_schedule_t       *schedule    = wrapper->schedule;
    ucc_status_t          status;

    /* Check if validation succeeded */
    if (!dt_check->validated) {
        /* Validation failed - propagate error to schedule and complete wrapper */
        wrapper->status = UCC_ERR_NOT_SUPPORTED;
        if (schedule) {
            schedule->super.status = UCC_ERR_NOT_SUPPORTED;
            schedule->super.super.status = UCC_ERR_NOT_SUPPORTED;
        }
        return UCC_OK;
    }
    /* Validation succeeded - post the actual task */
    status = actual_task->post(actual_task);
    if (status < 0) {
        wrapper->status = status;
        if (schedule) {
            schedule->super.status = status;
            schedule->super.super.status = status;
        }
        return status;
    }
    wrapper->status = UCC_INPROGRESS;
    return ucc_progress_queue_enqueue(wrapper->bargs.team->contexts[0]->pq, wrapper);
}

/**
 * Progress function for actual task wrapper
 *
 * Checks the actual task status. The actual task progresses itself via its own
 * progress queue enqueued by its post() function.
 */
static void ucc_dt_check_actual_wrapper_progress(ucc_coll_task_t *wrapper)
{
    ucc_dt_check_state_t *dt_check    = UCC_DT_CHECK_FROM_TASK(wrapper);
    ucc_coll_task_t      *actual_task = dt_check->actual_task;

    /* Just copy status from actual task to wrapper
     * The actual task progresses itself since it was enqueued by its post() */
    wrapper->status = actual_task->status;
}

/**
 * Finalize function for actual task wrapper
 */
static ucc_status_t ucc_dt_check_actual_wrapper_finalize(ucc_coll_task_t *wrapper)
{
    ucc_dt_check_state_t *dt_check    = UCC_DT_CHECK_FROM_TASK(wrapper);
    ucc_coll_task_t      *actual_task = dt_check->actual_task;
    ucc_status_t          status      = UCC_OK;

    /* Finalize the actual task */
    if (actual_task && actual_task->finalize) {
        status = actual_task->finalize(actual_task);
    }
    /* dt_check is embedded in schedule, will be freed with schedule */
    /* Wrapper is from memory pool */
    ucc_mpool_put(wrapper);
    return status;
}

/**
 * Finalize function for dt_check schedule
 *
 * Finalizes all tasks in the schedule and frees the schedule itself.
 */
static ucc_status_t ucc_dt_check_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_dt_check_schedule_t *dt_schedule =
        ucc_derived_of(task, ucc_dt_check_schedule_t);
    ucc_status_t             status;

    /* Finalize all tasks in the schedule */
    status = ucc_schedule_finalize(task);
    /* Destruct and free the schedule itself (including embedded dt_check) */
    ucc_coll_task_destruct(&dt_schedule->super.super);
    ucc_free(dt_schedule);
    return status;
}

ucc_coll_task_t* ucc_service_dt_check(ucc_team_t *team, ucc_coll_task_t *task)
{
    ucc_dt_check_schedule_t  *dt_schedule;
    ucc_dt_check_state_t     *dt_check;
    ucc_coll_task_t          *allreduce_wrapper;
    ucc_coll_task_t          *actual_wrapper;
    ucc_base_coll_args_t      empty_bargs;
    const ucc_coll_args_t    *args      = &task->bargs.args;
    ucc_rank_t                rank      = team->rank;
    int                       root      = (int)args->root;
    ucc_coll_type_t           coll_type = args->coll_type;
    ucc_datatype_t            local_dt       = UCC_DT_INT8;
    ucc_memory_type_t         local_mem_type = UCC_MEMORY_TYPE_UNKNOWN;
    int                       is_contig;
    ucc_status_t              status;

    /* If check is disabled, return original task */
    if (!ucc_global_config.check_asymmetric_dt) {
        return task;
    }

    /* Determine which datatype and memory type to check based on operation and rank */
    if (rank == root) {
        switch (coll_type) {
        case UCC_COLL_TYPE_GATHER:
            if (UCC_IS_INPLACE(*args)) {
                local_dt = args->dst.info.datatype;
                local_mem_type = args->dst.info.mem_type;
            } else {
                local_dt = args->src.info.datatype;
                local_mem_type = args->src.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_GATHERV:
            if (UCC_IS_INPLACE(*args)) {
                local_dt = args->dst.info_v.datatype;
                local_mem_type = args->dst.info_v.mem_type;
            } else {
                local_dt = args->src.info.datatype;
                local_mem_type = args->src.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_SCATTER:
            if (UCC_IS_INPLACE(*args)) {
                local_dt = args->src.info.datatype;
                local_mem_type = args->src.info.mem_type;
            } else {
                local_dt = args->dst.info.datatype;
                local_mem_type = args->dst.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_SCATTERV:
            if (UCC_IS_INPLACE(*args)) {
                local_dt = args->src.info_v.datatype;
                local_mem_type = args->src.info_v.mem_type;
            } else {
                local_dt = args->dst.info.datatype;
                local_mem_type = args->dst.info.mem_type;
            }
            break;
        default:
            /* Not a rooted collective, no validation needed */
            return task;
        }
    } else {
        switch (coll_type) {
        case UCC_COLL_TYPE_GATHER:
        case UCC_COLL_TYPE_GATHERV:
            local_dt = args->src.info.datatype;
            local_mem_type = args->src.info.mem_type;
            break;
        case UCC_COLL_TYPE_SCATTER:
        case UCC_COLL_TYPE_SCATTERV:
            local_dt = args->dst.info.datatype;
            local_mem_type = args->dst.info.mem_type;
            break;
        default:
            /* Not a rooted collective, no validation needed */
            return task;
        }
    }

    /* Determine if local datatype is contiguous */
    if (UCC_DT_IS_PREDEFINED(local_dt)) {
        is_contig = 1;
    } else if (UCC_DT_IS_GENERIC(local_dt)) {
        is_contig = UCC_DT_IS_CONTIG(local_dt);
    } else {
        is_contig = 0;
    }

    /* Allocate schedule with embedded dt_check */
    dt_schedule = (ucc_dt_check_schedule_t *)ucc_malloc(sizeof(*dt_schedule),
                                                         "dt_check_schedule");
    if (!dt_schedule) {
        ucc_error("failed to allocate dt_check_schedule");
        return NULL;
    }
    memset(dt_schedule, 0, sizeof(*dt_schedule));

    /* Initialize schedule with task's bargs and TL/CL team */
    ucc_coll_task_construct(&dt_schedule->super.super);
    status = ucc_schedule_init(&dt_schedule->super, &task->bargs, task->team);
    if (status != UCC_OK) {
        ucc_error("failed to initialize dt_check schedule: %s",
                  ucc_status_string(status));
        ucc_free(dt_schedule);
        return NULL;
    }

    /* Store actual task pointer */
    dt_schedule->actual_task = task;

    /* Setup embedded dt_check state */
    dt_check = &dt_schedule->dt_check;
    /* Setup values for min/max trick: [dt, -dt, mem, -mem] */
    if (!UCC_DT_IS_PREDEFINED(local_dt)) {
        /* Generic or invalid datatype - reject to prevent int16 overflow */
        dt_check->values[0] = (int16_t) UCC_ERR_NOT_SUPPORTED;
        dt_check->values[1] = -(int16_t) UCC_ERR_NOT_SUPPORTED;
    } else if (is_contig) {
        /* Predefined contiguous datatype - safe to cast to int16 */
        dt_check->values[0] = (int16_t) local_dt;
        dt_check->values[1] = -(int16_t) local_dt;
    } else {
        /* Predefined but non-contiguous datatype */
        dt_check->values[0] = (int16_t) UCC_ERR_NOT_SUPPORTED;
        dt_check->values[1] = -(int16_t) UCC_ERR_NOT_SUPPORTED;
    }
    dt_check->values[2] = (int16_t) local_mem_type;
    dt_check->values[3] = -(int16_t) local_mem_type;
    /* Setup subset for full team */
    dt_check->subset.myrank = team->rank;
    dt_check->subset.map.type = UCC_EP_MAP_FULL;
    dt_check->subset.map.ep_num = team->size;
    /* Initialize check_req to NULL */
    dt_check->check_req = NULL;
    dt_check->validated = 0;
    dt_check->actual_task = task;

    /* Create allreduce wrapper task from memory pool */
    allreduce_wrapper = ucc_mpool_get(&task->bargs.team->contexts[0]->lib->stub_tasks_mp);
    if (!allreduce_wrapper) {
        ucc_error("failed to allocate allreduce wrapper task from mpool");
        goto error_schedule;
    }
    /* Initialize allreduce wrapper task with minimal bargs (only needs team) */
    memset(&empty_bargs, 0, sizeof(empty_bargs));
    empty_bargs.team = task->bargs.team;
    status = ucc_coll_task_init(allreduce_wrapper, &empty_bargs, task->team);
    if (status != UCC_OK) {
        ucc_error("failed to init allreduce wrapper task: %s",
                  ucc_status_string(status));
        ucc_mpool_put(allreduce_wrapper);
        goto error_schedule;
    }
    /* Set allreduce wrapper functions */
    allreduce_wrapper->post     = ucc_dt_check_allreduce_post;
    allreduce_wrapper->progress = ucc_dt_check_allreduce_progress;
    allreduce_wrapper->finalize = ucc_dt_check_allreduce_finalize;
    allreduce_wrapper->status   = UCC_OPERATION_INITIALIZED;
    /* Add allreduce wrapper to schedule - starts when schedule starts */
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

    /* Create actual task wrapper that conditionally posts the actual task */
    actual_wrapper = ucc_mpool_get(&task->bargs.team->contexts[0]->lib->stub_tasks_mp);
    if (!actual_wrapper) {
        ucc_error("failed to allocate actual wrapper task from mpool");
        goto error_allreduce_wrapper;
    }
    status = ucc_coll_task_init(actual_wrapper, &empty_bargs, task->team);
    if (status != UCC_OK) {
        ucc_error("failed to init actual wrapper task: %s",
                  ucc_status_string(status));
        ucc_mpool_put(actual_wrapper);
        goto error_allreduce_wrapper;
    }
    /* Set actual wrapper functions */
    actual_wrapper->post     = ucc_dt_check_actual_wrapper_post;
    actual_wrapper->progress = ucc_dt_check_actual_wrapper_progress;
    actual_wrapper->finalize = ucc_dt_check_actual_wrapper_finalize;
    actual_wrapper->status   = UCC_OPERATION_INITIALIZED;
    /* Add actual wrapper to schedule - depends on allreduce completing */
    status = ucc_task_subscribe_dep(allreduce_wrapper, actual_wrapper, UCC_EVENT_COMPLETED);
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

    /* Set schedule functions */
    dt_schedule->super.super.post     = ucc_schedule_start;
    dt_schedule->super.super.progress = NULL; /* Sub-tasks have their own progress functions */
    dt_schedule->super.super.finalize = ucc_dt_check_schedule_finalize;
    return &dt_schedule->super.super;

error_allreduce_wrapper:
    ucc_mpool_put(allreduce_wrapper);
error_schedule:
    ucc_coll_task_destruct(&dt_schedule->super.super);
    ucc_free(dt_schedule);
    return NULL;
}
