/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2022.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm_executor.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_atomic.h"

ucc_status_t ucc_rocm_executor_interruptible_get_stream(hipStream_t *stream)
{
    static uint32_t last_used   = 0;
    int             num_streams = EC_ROCM_CONFIG->exec_num_streams;
    ucc_status_t    st;
    int             i, j;
    uint32_t        id;

    if (ucc_unlikely(!ucc_ec_rocm.exec_streams_initialized)) {
        ucc_spin_lock(&ucc_ec_rocm.init_spinlock);
        if (ucc_ec_rocm.exec_streams_initialized) {
            goto unlock;
        }

        for(i = 0; i < num_streams; i++) {
            st = ROCM_FUNC(hipStreamCreateWithFlags(&ucc_ec_rocm.exec_streams[i],
                                                     hipStreamNonBlocking));
            if (st != UCC_OK) {
                for (j = 0; j < i; j++) {
                    ROCM_FUNC(hipStreamDestroy(ucc_ec_rocm.exec_streams[j]));
                }
                ucc_spin_unlock(&ucc_ec_rocm.init_spinlock);
                return st;
            }
        }
        ucc_ec_rocm.exec_streams_initialized = 1;
unlock:
        ucc_spin_unlock(&ucc_ec_rocm.init_spinlock);
    }

    id = ucc_atomic_fadd32(&last_used, 1);
    *stream = ucc_ec_rocm.exec_streams[id % num_streams];
    return UCC_OK;
}

ucc_status_t
ucc_rocm_executor_interruptible_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    ucc_ec_rocm_executor_interruptible_task_t *ee_task;
    hipStream_t stream;
    ucc_status_t status;

    status = ucc_rocm_executor_interruptible_get_stream(&stream);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    ee_task = ucc_mpool_get(&ucc_ec_rocm.executor_interruptible_tasks);
    if (ucc_unlikely(!ee_task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status  = ucc_ec_rocm_event_create(&ee_task->event);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_mpool_put(ee_task);
        return status;
    }

    ee_task->super.status = UCC_INPROGRESS;
    ee_task->super.eee    = executor;
    memcpy(&ee_task->super.args, task_args, sizeof(ucc_ee_executor_task_args_t));
    switch (task_args->task_type) {
    case UCC_EE_EXECUTOR_TASK_TYPE_COPY:
        status = ROCM_FUNC(hipMemcpyAsync(task_args->bufs[0],
                                          task_args->bufs[1],
                                          task_args->count, hipMemcpyDefault,
                                          stream));
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start memcpy op");
            goto free_task;
        }

        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE:
        /* temp workaround to avoid code duplication*/
        status = ucc_mc_reduce(task_args->bufs[1], task_args->bufs[2],
                               task_args->bufs[0], task_args->count,
                               task_args->dt, task_args->op,
                               UCC_MEMORY_TYPE_ROCM);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start reduce op");
            goto free_task;
        }

        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI:
        status = ucc_mc_reduce_multi(task_args->bufs[1], task_args->bufs[2],
                                     task_args->bufs[0], task_args->size,
                                     task_args->count, task_args->stride,
                                     task_args->dt, task_args->op,
                                     UCC_MEMORY_TYPE_ROCM);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start reduce multi op");
            goto free_task;
        }
        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI_ALPHA:
        status = ucc_mc_reduce_multi_alpha(task_args->bufs[1],
                                           task_args->bufs[2],
                                           task_args->bufs[0],
                                           task_args->size, task_args->count,
                                           task_args->stride, task_args->dt,
                                           task_args->op, UCC_OP_PROD,
                                           task_args->alpha,
                                           UCC_MEMORY_TYPE_ROCM);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start reduce multi alpha op");
            goto free_task;
        }
        break;
    default:
        ec_error(&ucc_ec_rocm.super, "executor operation is not supported task_type %d", task_args->task_type);
        status = UCC_ERR_INVALID_PARAM;
        goto free_task;
    }

    status = ucc_ec_rocm_event_post(stream, ee_task->event);
    if (ucc_unlikely(status != UCC_OK)) {
        goto free_task;
    }

    *task = &ee_task->super;
    return UCC_OK;

free_task:
    ucc_ec_rocm_event_destroy(ee_task->event);
    ucc_mpool_put(ee_task);
    return status;
}

ucc_status_t
ucc_rocm_executor_interruptible_task_test(const ucc_ee_executor_task_t *task)
{
    ucc_ec_rocm_executor_interruptible_task_t *ee_task =
        ucc_derived_of(task, ucc_ec_rocm_executor_interruptible_task_t);

    ee_task->super.status = ucc_ec_rocm_event_test(ee_task->event);
    return ee_task->super.status;
}

ucc_status_t
ucc_rocm_executor_interruptible_task_finalize(ucc_ee_executor_task_t *task)
{
    ucc_ec_rocm_executor_interruptible_task_t *ee_task =
        ucc_derived_of(task, ucc_ec_rocm_executor_interruptible_task_t);
    ucc_status_t status;

    status = ucc_ec_rocm_event_destroy(ee_task->event);
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_rocm_executor_interruptible_start(ucc_ee_executor_t *executor)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_rocm_executor_t);

    eee->mode  = UCC_EC_ROCM_EXECUTOR_MODE_INTERRUPTIBLE;
    eee->state = UCC_EC_ROCM_EXECUTOR_STARTED;

    eee->ops.task_post     = ucc_rocm_executor_interruptible_task_post;
    eee->ops.task_test     = ucc_rocm_executor_interruptible_task_test;
    eee->ops.task_finalize = ucc_rocm_executor_interruptible_task_finalize;

    return UCC_OK;
}

ucc_status_t ucc_rocm_executor_interruptible_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_rocm_executor_t);

    eee->state = UCC_EC_ROCM_EXECUTOR_INITIALIZED;
    return UCC_OK;
}
