/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda_executor.h"
#include "utils/arch/cpu.h"
#include "components/mc/ucc_mc.h"

ucc_status_t ucc_cuda_executor_init(const ucc_ee_executor_params_t *params,
                                    ucc_ee_executor_t **executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_mpool_get(&ucc_ec_cuda.executors);

    if (ucc_unlikely(!eee)) {
        ec_error(&ucc_ec_cuda.super, "failed to allocate executor");
        return UCC_ERR_NO_MEMORY;
    }
    UCC_EC_CUDA_INIT_STREAM();
    ec_debug(&ucc_ec_cuda.super, "executor init, eee: %p", eee);
    eee->super.ee_type = params->ee_type;
    eee->state         = UCC_EC_CUDA_EXECUTOR_INITIALIZED;

    *executor = &eee->super;
    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_status(const ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);

    switch (eee->state) {
    case UCC_EC_CUDA_EXECUTOR_INITIALIZED:
        return UCC_OPERATION_INITIALIZED;
    case UCC_EC_CUDA_EXECUTOR_POSTED:
        return UCC_INPROGRESS;
    case UCC_EC_CUDA_EXECUTOR_STARTED:
        return UCC_OK;
    default:
/* executor has been destroyed */
        return UCC_ERR_NO_RESOURCE;
    }
}

ucc_status_t ucc_cuda_executor_finalize(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);

    ec_debug(&ucc_ec_cuda.super, "executor free, eee: %p", eee);
    ucc_assert(eee->state == UCC_EC_CUDA_EXECUTOR_INITIALIZED);
    ucc_mpool_put(eee);

    return UCC_OK;
}

ucc_status_t
ucc_cuda_executor_interruptible_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    ucc_ec_cuda_executor_interruptible_task_t *ee_task;
    ucc_status_t status;

    ee_task = ucc_mpool_get(&ucc_ec_cuda.executor_interruptible_tasks);
    if (ucc_unlikely(!ee_task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status  = ucc_ec_cuda_event_create(&ee_task->event);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_mpool_put(ee_task);
        return status;
    }
    ee_task->super.status = UCC_INPROGRESS;
    ee_task->super.eee    = executor;
    memcpy(&ee_task->super.args, task_args, sizeof(ucc_ee_executor_task_args_t));
    switch (task_args->task_type) {
    case UCC_EE_EXECUTOR_TASK_TYPE_COPY:
        status = CUDA_FUNC(cudaMemcpyAsync(task_args->bufs[0],
                                           task_args->bufs[1],
                                           task_args->count, cudaMemcpyDefault,
                                           ucc_ec_cuda.stream));
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start memcpy op");
            goto free_task;
        }

        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE:
        /* temp workaround to avoid code duplication*/
        status = ucc_mc_reduce(task_args->bufs[1], task_args->bufs[2],
                               task_args->bufs[0], task_args->count,
                               task_args->dt, task_args->op,
                               UCC_MEMORY_TYPE_CUDA);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start reduce op");
            goto free_task;
        }

        break;
    default:
        ec_error(&ucc_ec_cuda.super, "executor operation is not supported");
        status = UCC_ERR_INVALID_PARAM;
        goto free_task;
    }

    status = ucc_ec_cuda_event_post(ucc_ec_cuda.stream, ee_task->event);
    if (ucc_unlikely(status != UCC_OK)) {
        goto free_task;
    }

    *task = &ee_task->super;
    return UCC_OK;

free_task:
    ucc_ec_cuda_event_destroy(ee_task->event);
    ucc_mpool_put(ee_task);
    return status;
}

ucc_status_t
ucc_cuda_executor_interruptible_task_test(const ucc_ee_executor_task_t *task)
{
    ucc_ec_cuda_executor_interruptible_task_t *ee_task =
        ucc_derived_of(task, ucc_ec_cuda_executor_interruptible_task_t);

    ee_task->super.status = ucc_ec_cuda_event_test(ee_task->event);
    return ee_task->super.status;
}

ucc_status_t
ucc_cuda_executor_interruptible_task_finalize(ucc_ee_executor_task_t *task)
{
    ucc_ec_cuda_executor_interruptible_task_t *ee_task =
        ucc_derived_of(task, ucc_ec_cuda_executor_interruptible_task_t);
    ucc_status_t status;

    status = ucc_ec_cuda_event_destroy(ee_task->event);
    ucc_mpool_put(task);
    return status;
}

ucc_status_t
ucc_cuda_executor_persistent_task_post(ucc_ee_executor_t *executor,
                                       const ucc_ee_executor_task_args_t *task_args,
                                       ucc_ee_executor_task_t **task)
{
    ucc_ec_cuda_executor_t *eee       = ucc_derived_of(executor,
                                                       ucc_ec_cuda_executor_t);
    int                     max_tasks = EC_CUDA_CONFIG->exec_max_tasks;
    ucc_ee_executor_task_t *ee_task;

    if (task_args->task_type == UCC_EE_EXECUTOR_TASK_TYPE_REDUCE) {
        if (task_args->op != UCC_OP_SUM) {
            return UCC_ERR_NOT_SUPPORTED;
        }
        if ((task_args->dt != UCC_DT_FLOAT32) &&
            (task_args->dt != UCC_DT_FLOAT64) &&
            (task_args->dt != UCC_DT_INT32)) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }
    if (ucc_ec_cuda.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spin_lock(&eee->tasks_lock);
    }
    ee_task         = &(eee->tasks[eee->pidx % max_tasks]);
    ee_task->eee    = executor;
    ee_task->status = UCC_OPERATION_INITIALIZED;
    memcpy(&ee_task->args, task_args, sizeof(ucc_ee_executor_task_args_t));
    ucc_memory_bus_fence();
    eee->pidx += 1;
    if (ucc_ec_cuda.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spin_unlock(&eee->tasks_lock);
    }
    ec_debug(&ucc_ec_cuda.super, "executor task post, eee: %p", eee);

    *task = ee_task;
    return UCC_OK;
}


ucc_status_t
ucc_cuda_executor_persistent_task_test(const ucc_ee_executor_task_t *task)
{
    CUDA_CHECK(cudaGetLastError());
    return task->status;
}

ucc_status_t
ucc_cuda_executor_persistent_task_finalize(ucc_ee_executor_task_t *task)
{
    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    return eee->ops.task_post(executor, task_args, task);
}


ucc_status_t ucc_cuda_executor_task_test(const ucc_ee_executor_task_t *task)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(task->eee,
                                                 ucc_ec_cuda_executor_t);
    return eee->ops.task_test(task);
}

ucc_status_t ucc_cuda_executor_task_finalize(ucc_ee_executor_task_t *task)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(task->eee,
                                                 ucc_ec_cuda_executor_t);
    return eee->ops.task_finalize(task);
}

ucc_status_t ucc_cuda_executor_interruptible_start(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);

    eee->mode  = UCC_EC_CUDA_EXECUTOR_MODE_INTERRUPTIBLE;
    eee->state = UCC_EC_CUDA_EXECUTOR_STARTED;

    eee->ops.task_post     = ucc_cuda_executor_interruptible_task_post;
    eee->ops.task_test     = ucc_cuda_executor_interruptible_task_test;
    eee->ops.task_finalize = ucc_cuda_executor_interruptible_task_finalize;

    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_persistent_start(ucc_ee_executor_t *executor,
                                                void *ee_context)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    ucc_status_t status;

    ucc_assert(eee->state == UCC_EC_CUDA_EXECUTOR_INITIALIZED);
    ec_debug(&ucc_ec_cuda.super, "executor start, eee: %p", eee);
    eee->super.ee_context = ee_context;
    eee->state            = UCC_EC_CUDA_EXECUTOR_POSTED;
    eee->pidx             = 0;
    eee->mode             = UCC_EC_CUDA_EXECUTOR_MODE_PERSISTENT;

    status = ucc_ec_cuda_persistent_kernel_start(eee);
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cuda.super, "failed to launch executor kernel");
        return status;
    }

    eee->ops.task_post     = ucc_cuda_executor_persistent_task_post;
    eee->ops.task_test     = ucc_cuda_executor_persistent_task_test;
    eee->ops.task_finalize = ucc_cuda_executor_persistent_task_finalize;
    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_start(ucc_ee_executor_t *executor,
                                     void *ee_context)
{
    if (!ee_context) {
        return ucc_cuda_executor_interruptible_start(executor);
    } else {
        return ucc_cuda_executor_persistent_start(executor, ee_context);
    }
}

ucc_status_t ucc_cuda_executor_interruptible_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);

    eee->state = UCC_EC_CUDA_EXECUTOR_INITIALIZED;
    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_persistent_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    volatile ucc_ec_cuda_executor_state_t *st = &eee->state;

    ec_debug(&ucc_ec_cuda.super, "executor stop, eee: %p", eee);
    /* can be safely ended only if it's in STARTED or COMPLETED_ACK state */
    ucc_assert((*st != UCC_EC_CUDA_EXECUTOR_POSTED) &&
               (*st != UCC_EC_CUDA_EXECUTOR_SHUTDOWN));
    *st = UCC_EC_CUDA_EXECUTOR_SHUTDOWN;
    eee->pidx = -1;
    while(*st != UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK) { }
    eee->super.ee_context = NULL;
    eee->state = UCC_EC_CUDA_EXECUTOR_INITIALIZED;

    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    if (eee->mode == UCC_EC_CUDA_EXECUTOR_MODE_INTERRUPTIBLE) {
        return ucc_cuda_executor_interruptible_stop(executor);
    } else {
        return ucc_cuda_executor_persistent_stop(executor);
    }
}
