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
#include "ucc_global_opts.h"
#include "ucc_service_coll.h"


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

/**
 * Start non-blocking datatype validation for rooted collective
 *
 * This function extracts datatypes, allocates buffers, and prepares the
 * dt_check state for validation. Called during collective init.
 */
/**
 * Start asymmetric datatype validation using service allreduce
 *
 * Prepares dt_check state for validation but doesn't start allreduce yet.
 * The allreduce will be started by the allreduce wrapper task's post function.
 *
 * Uses min/max trick: send [dt, -dt, mem, -mem] with MIN reduction.
 * After reduction: if min(dt) == -min(-dt), all ranks have same datatype.
 */
static ucc_status_t ucc_dt_check_start_validation(ucc_coll_task_t *task)
{
    ucc_dt_check_state_t  *dt_check = NULL;
    ucc_status_t           status   = UCC_OK;
    ucc_datatype_t         local_dt = UCC_DT_INT8;
    ucc_memory_type_t      local_mem_type = UCC_MEMORY_TYPE_UNKNOWN;
    const ucc_coll_args_t *args      = &task->bargs.args;
    ucc_team_t            *team      = task->bargs.team;
    ucc_rank_t             rank      = team->rank;
    int                    root      = (int)args->root;
    ucc_coll_type_t        coll_type = args->coll_type;
    int                    is_contig;

    /* If check is disabled, skip validation */
    if (!ucc_global_config.check_asymmetric_dt) {
        task->dt_check = NULL;
        return UCC_OK;
    }
    /* Determine which datatype and memory type to check based on operation and rank */
    if (rank == root) {
        switch (coll_type) {
        case UCC_COLL_TYPE_GATHER:
        case UCC_COLL_TYPE_GATHERV:
            if (UCC_IS_INPLACE(*args)) {
                local_dt = args->dst.info.datatype;
                local_mem_type = args->dst.info.mem_type;
            } else {
                local_dt = args->src.info.datatype;
                local_mem_type = args->src.info.mem_type;
            }
            break;
        case UCC_COLL_TYPE_SCATTER:
        case UCC_COLL_TYPE_SCATTERV:
            if (UCC_IS_INPLACE(*args)) {
                local_dt = args->src.info.datatype;
                local_mem_type = args->src.info.mem_type;
            } else {
                local_dt = args->dst.info.datatype;
                local_mem_type = args->dst.info.mem_type;
            }
            break;
        default:
            goto out;
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
            goto out;
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
    /* Allocate dt_check state */
    dt_check = (ucc_dt_check_state_t *)ucc_malloc(sizeof(ucc_dt_check_state_t),
                                                   "dt_check_state");
    if (!dt_check) {
        ucc_error("Failed to allocate dt_check state");
        return UCC_ERR_NO_MEMORY;
    }
    memset(dt_check, 0, sizeof(ucc_dt_check_state_t));
    task->dt_check = dt_check;
    /* Setup local values for min/max trick: [dt, -dt, mem, -mem]
     * Note: Only predefined datatypes are supported (max value 136 fits in int16).
     * Generic datatypes use 64-bit pointers and would overflow int16, so we
     * explicitly reject them here to prevent overflow from user arguments. */
    if (!UCC_DT_IS_PREDEFINED(local_dt)) {
        /* Generic or invalid datatype - reject to prevent int16 overflow */
        dt_check->local_values[0] = (int16_t) UCC_ERR_NOT_SUPPORTED;
        dt_check->local_values[1] = -(int16_t) UCC_ERR_NOT_SUPPORTED;
    } else if (is_contig) {
        /* Predefined contiguous datatype - safe to cast to int16 */
        dt_check->local_values[0] = (int16_t) local_dt;
        dt_check->local_values[1] = -(int16_t) local_dt;
    } else {
        /* Predefined but non-contiguous datatype */
        dt_check->local_values[0] = (int16_t) UCC_ERR_NOT_SUPPORTED;
        dt_check->local_values[1] = -(int16_t) UCC_ERR_NOT_SUPPORTED;
    }
    dt_check->local_values[2] = (int16_t) local_mem_type;
    dt_check->local_values[3] = -(int16_t) local_mem_type;
    /* Setup subset for full team - will be used when service allreduce is posted */
    dt_check->subset.myrank = team->rank;
    dt_check->subset.map.type = UCC_EP_MAP_FULL;
    dt_check->subset.map.ep_num = team->size;
    /* Initialize check_req to NULL - will be created in allreduce_post */
    dt_check->check_req = NULL;
    dt_check->validated = 0;
    return UCC_OK;

out:
    if (dt_check) {
        ucc_free(dt_check);
        task->dt_check = NULL;
    }
    return status;
}

/**
 * Validate allreduced datatype values using min/max trick
 *
 * After MIN allreduce on [dt, -dt, mem, -mem]:
 *   - reduced[0] contains min(dt) across all ranks
 *   - reduced[1] contains min(-dt) = -max(dt) across all ranks
 *   - If reduced[0] == -reduced[1], all ranks have identical dt
 *   - Same logic applies to memory type with reduced[2] and reduced[3]
 */
static ucc_status_t ucc_dt_validate_results(ucc_coll_task_t *task)
{
    ucc_dt_check_state_t *dt_check = task->dt_check;
    int16_t              *reduced;

    /* Safety checks */
    if (!dt_check) {
        return UCC_ERR_INVALID_PARAM;
    }
    reduced = dt_check->reduced_values;
    /* Check if any rank has non-contiguous datatype */
    if (reduced[0] == (int16_t) UCC_ERR_NOT_SUPPORTED) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    /* Check if all ranks have the same datatype using min/max trick */
    if (reduced[0] != -reduced[1]) {
        return UCC_ERR_INVALID_PARAM;
    }
    /* Check if all ranks have the same memory type */
    if (reduced[2] != -reduced[3]) {
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
    ucc_dt_check_state_t *dt_check = allreduce_wrapper->dt_check;
    ucc_team_t           *team = allreduce_wrapper->bargs.team;
    ucc_status_t          status;

    /* Safety check */
    if (!dt_check) {
        allreduce_wrapper->status = UCC_ERR_INVALID_PARAM;
        return UCC_ERR_INVALID_PARAM;
    }

    /* Start service allreduce with MIN operation on 4 int16_t values */
    status = ucc_service_allreduce(team, dt_check->local_values,
                                   dt_check->reduced_values,
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
    ucc_dt_check_state_t *dt_check = allreduce_wrapper->dt_check;
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
    status = ucc_dt_validate_results(allreduce_wrapper);
    dt_check->validated = (status == UCC_OK);
    /* Completes with UCC_OK so schedule continues to actual wrapper */
    allreduce_wrapper->status = UCC_OK;
}

/**
 * Finalize function for allreduce wrapper task
 */
static ucc_status_t ucc_dt_check_allreduce_finalize(ucc_coll_task_t *allreduce_wrapper)
{
    ucc_dt_check_state_t *dt_check = allreduce_wrapper->dt_check;

    /* Clean up check_req if it wasn't finalized in progress */
    if (dt_check && dt_check->check_req) {
        ucc_service_coll_finalize(dt_check->check_req);
        dt_check->check_req = NULL;
    }
    /* dt_check is shared with actual wrapper, so don't free it here */
    allreduce_wrapper->dt_check = NULL;
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
    ucc_dt_check_state_t *dt_check = wrapper->dt_check;
    ucc_coll_task_t      *actual_task = dt_check->actual_task;
    ucc_schedule_t       *schedule = wrapper->schedule;
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
    ucc_dt_check_state_t *dt_check = wrapper->dt_check;
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
    ucc_dt_check_state_t *dt_check = wrapper->dt_check;
    ucc_coll_task_t      *actual_task = dt_check->actual_task;
    ucc_status_t          status = UCC_OK;

    /* Finalize the actual task */
    if (actual_task && actual_task->finalize) {
        status = actual_task->finalize(actual_task);
    }
    /* Clean up dt_check state (deferred from validation finalize) */
    if (dt_check) {
        ucc_free(dt_check);
    }
    wrapper->dt_check = NULL;
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
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    /* Finalize all tasks in the schedule */
    status = ucc_schedule_finalize(task);
    /* Destruct and free the schedule itself */
    ucc_coll_task_destruct(&schedule->super);
    ucc_free(schedule);
    return status;
}

/**
 * Create a schedule with service allreduce and actual collective
 *
 * Creates a schedule containing two tasks:
 *   1. Allreduce wrapper task - validates datatype consistency using MIN reduction
 *   2. Actual wrapper task - conditionally posts the real gather/scatter operation
 *
 * Dependencies: allreduce wrapper → actual wrapper
 *
 * The allreduce wrapper uses the internal service allreduce API with MIN operation
 * and the min/max trick to detect mismatches in O(1) message size (doesn't scale with ranks).
 * The actual wrapper checks the validation result and only posts the real collective
 * if validation succeeded.
 *
 * @param task The actual collective task (already created by TL/CL)
 * @return Pointer to schedule (as ucc_coll_task_t*), or NULL on error
 */
static ucc_coll_task_t* ucc_dt_check_create_schedule(ucc_coll_task_t *task)
{
    ucc_schedule_t           *schedule;
    ucc_coll_task_t          *allreduce_wrapper;
    ucc_coll_task_t          *actual_wrapper;
    ucc_base_coll_args_t      empty_bargs;
    ucc_status_t              status;

    /* Allocate schedule */
    schedule = (ucc_schedule_t *)ucc_malloc(sizeof(*schedule), "dt_check_schedule");
    if (!schedule) {
        ucc_error("failed to allocate schedule for dt_check");
        return NULL;
    }
    /* Initialize schedule with task's bargs and TL/CL team */
    ucc_coll_task_construct(&schedule->super);
    status = ucc_schedule_init(schedule, &task->bargs, task->team);
    if (status != UCC_OK) {
        ucc_error("failed to initialize dt_check schedule: %s",
                  ucc_status_string(status));
        ucc_free(schedule);
        return NULL;
    }
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
    /* Give dt_check state to allreduce wrapper */
    allreduce_wrapper->dt_check = task->dt_check;
    /* Store actual_task pointer in dt_check for wrapper to access */
    allreduce_wrapper->dt_check->actual_task = task;
    /* Set allreduce wrapper functions */
    allreduce_wrapper->post     = ucc_dt_check_allreduce_post;
    allreduce_wrapper->progress = ucc_dt_check_allreduce_progress;
    allreduce_wrapper->finalize = ucc_dt_check_allreduce_finalize;
    allreduce_wrapper->status   = UCC_OPERATION_INITIALIZED;
    /* Add allreduce wrapper to schedule - starts when schedule starts */
    status = ucc_task_subscribe_dep(&schedule->super, allreduce_wrapper,
                                    UCC_EVENT_SCHEDULE_STARTED);
    if (status != UCC_OK) {
        ucc_error("failed to subscribe allreduce wrapper: %s",
                  ucc_status_string(status));
        goto error_allreduce_wrapper;
    }
    status = ucc_schedule_add_task(schedule, allreduce_wrapper);
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
    /* Share dt_check with actual wrapper - it contains pointer to actual_task */
    actual_wrapper->dt_check = allreduce_wrapper->dt_check;
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
    status = ucc_schedule_add_task(schedule, actual_wrapper);
    if (status != UCC_OK) {
        ucc_error("failed to add actual wrapper to schedule: %s",
                  ucc_status_string(status));
        ucc_mpool_put(actual_wrapper);
        goto error_allreduce_wrapper;
    }
    /* Set schedule functions */
    schedule->super.post     = ucc_schedule_start;
    schedule->super.progress = NULL; /* Sub-tasks have their own progress functions */
    schedule->super.finalize = ucc_dt_check_schedule_finalize;
    return &schedule->super;

error_allreduce_wrapper:
    ucc_mpool_put(allreduce_wrapper);  /* Return to pool, not free */
error_schedule:
    ucc_coll_task_destruct(&schedule->super);
    ucc_free(schedule);
    /* Note: dt_check is still attached to task, no need to restore */
    return NULL;
}

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

    /* Setup non-blocking datatype check for rooted collectives
     *
     * This implements transparent validation using a schedule with two tasks:
     * 1. Allreduce validation task: uses MIN reduction with min/max trick to detect mismatches
     * 2. Actual collective task: the real gather/scatter operation
     *
     * Validation uses allreduce (MIN) on [dt, -dt, mem, -mem]:
     *   - Message size: 8 bytes (4 × int16_t, doesn't scale with number of ranks)
     *   - After reduction: min(dt) == -min(-dt) means all ranks have same datatype
     *
     * Dependencies: allreduce validation → actual task
     * If validation fails, the dependency mechanism prevents the actual task from posting.
     */
    if (coll_args->coll_type == UCC_COLL_TYPE_GATHER ||
        coll_args->coll_type == UCC_COLL_TYPE_GATHERV ||
        coll_args->coll_type == UCC_COLL_TYPE_SCATTER ||
        coll_args->coll_type == UCC_COLL_TYPE_SCATTERV) {
        /* First check if validation is needed */
        status = ucc_dt_check_start_validation(task);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_error("datatype check initialization failed");
            goto coll_finalize;
        }
        /* If validation is enabled, create schedule */
        if (task->dt_check) {
            ucc_coll_task_t *schedule;

            schedule = ucc_dt_check_create_schedule(task);
            if (!schedule) {
                ucc_error("failed to create dt_check schedule");
                /* Clean up dt_check state before going to coll_finalize */
                if (task->dt_check) {
                    ucc_free(task->dt_check);
                    task->dt_check = NULL;
                }
                status = UCC_ERR_NO_MEMORY;
                goto coll_finalize;
            }
            /* Return schedule to user instead of actual task */
            task = schedule;
        }
    }
    /* Setup top-level task (actual task or schedule) */
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


