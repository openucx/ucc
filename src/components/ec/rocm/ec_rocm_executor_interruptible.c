/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <stdbool.h>
#include "ec_rocm_executor.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "utils/ucc_atomic.h"

static bool ucc_ec_rocm_copy_multi_use_host (const ucc_ee_executor_task_args_t* task_args)
{
    bool result = true;

    for (int i = 0; i < task_args->copy_multi.num_vectors; i++) {
        if (task_args->copy_multi.counts[i] > EC_ROCM_CONFIG->copy_host_limit) {
            result = false;
            break;
        }
    }

    return result;
}

static int ucc_ec_rocm_total_reduce_len(const ucc_ee_executor_task_args_t* task_args)
{
    int             total_len = 0;
    ucc_datatype_t  dt;
    size_t          count;

    if (task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        dt    = task_args->reduce.dt;
        count = task_args->reduce.count;
    } else {
        ucc_assert(task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED);
        dt    = task_args->reduce_strided.dt;
        count = task_args->reduce_strided.count;
    }
    total_len += count * ucc_dt_size(dt);

    return total_len;
}

static bool ucc_ec_rocm_host_dt_supported(const ucc_ee_executor_task_args_t*  task_args)
{
    bool            result = false;
    ucc_datatype_t  dt;

    if (task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        dt     = task_args->reduce.dt;
    } else {
        ucc_assert(task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED);
        dt     = task_args->reduce_strided.dt;
    }
    if (dt != UCC_DT_BFLOAT16        &&
        dt != UCC_DT_FLOAT16         &&
        dt != UCC_DT_FLOAT32_COMPLEX &&
        dt != UCC_DT_FLOAT64_COMPLEX) {
        result = true;
    }
    return result;
}

static inline
bool ec_rocm_use_host_ops(const ucc_ee_executor_task_args_t *_task_args)
{
    if ( (_task_args->task_type == UCC_EE_EXECUTOR_TASK_COPY                            &&
          _task_args->copy.len <= EC_ROCM_CONFIG->copy_host_limit)                      ||
         (_task_args->task_type == UCC_EE_EXECUTOR_TASK_COPY_MULTI                      &&
          ucc_ec_rocm_copy_multi_use_host(_task_args))                                  ||
         ((_task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE                         ||
           _task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED)                &&
          ucc_ec_rocm_total_reduce_len(_task_args) <= EC_ROCM_CONFIG->reduce_host_limit &&
          ucc_ec_rocm_host_dt_supported(_task_args) )) {
        return true;
    }
    return false;
}

ucc_status_t ucc_rocm_executor_interruptible_get_stream(hipStream_t *stream)
{
    static uint32_t last_used   = 0;
    int             num_streams = EC_ROCM_CONFIG->exec_num_streams;
    ucc_status_t    st;
    int             i, j;
    uint32_t        id;

    ucc_assert(num_streams > 0);
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

ucc_status_t ucc_ec_rocm_copy_multi_kernel(const ucc_ee_executor_task_args_t *args,
                                           hipStream_t stream);

ucc_status_t
ucc_rocm_executor_interruptible_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    ucc_ec_rocm_executor_interruptible_task_t *ee_task;
    hipStream_t stream;
    ucc_status_t status;

    if (ec_rocm_use_host_ops(task_args)) {
        status = ucc_ee_executor_task_post(ucc_ec_rocm.cpu_executor, task_args,
                                           task);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to execute host ops from ROCm component");
        }
        return status;
    }

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
    case UCC_EE_EXECUTOR_TASK_COPY:
        status = ROCM_FUNC(hipMemcpyAsync(task_args->copy.dst,
                                          task_args->copy.src,
                                          task_args->copy.len, hipMemcpyDefault,
                                          stream));
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start memcpy op");
            goto free_task;
        }
      break;
    case UCC_EE_EXECUTOR_TASK_COPY_MULTI:
        status = ucc_ec_rocm_copy_multi_kernel(task_args, stream);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start copy multi op");
            goto free_task;
        }
        break;
    case UCC_EE_EXECUTOR_TASK_REDUCE:
    case UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED:
        status = ucc_ec_rocm_reduce((ucc_ee_executor_task_args_t *)task_args,
                                    stream);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_rocm.super, "failed to start reduce op");
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
