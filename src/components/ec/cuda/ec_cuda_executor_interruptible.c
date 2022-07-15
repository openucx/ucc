/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda_executor.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_atomic.h"

ucc_status_t ucc_cuda_executor_interruptible_get_stream(cudaStream_t *stream)
{
    static uint32_t last_used   = 0;
    int             num_streams = EC_CUDA_CONFIG->exec_num_streams;
    ucc_status_t    st;
    int             i, j;
    uint32_t        id;

    if (ucc_unlikely(!ucc_ec_cuda.exec_streams_initialized)) {
        ucc_spin_lock(&ucc_ec_cuda.init_spinlock);
        if (ucc_ec_cuda.exec_streams_initialized) {
            goto unlock;
        }

        for(i = 0; i < num_streams; i++) {
            st = CUDA_FUNC(cudaStreamCreateWithFlags(&ucc_ec_cuda.exec_streams[i],
                                                     cudaStreamNonBlocking));
            if (st != UCC_OK) {
                for (j = 0; j < i; j++) {
                    CUDA_FUNC(cudaStreamDestroy(ucc_ec_cuda.exec_streams[j]));
                }
                ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
                return st;
            }
        }
        ucc_ec_cuda.exec_streams_initialized = 1;
unlock:
        ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
    }

    id = ucc_atomic_fadd32(&last_used, 1);
    *stream = ucc_ec_cuda.exec_streams[id % num_streams];
    return UCC_OK;
}


ucc_status_t ucc_ec_cuda_copy_multi_kernel(const ucc_ee_executor_task_args_t *args,
                                           cudaStream_t stream);

ucc_status_t
ucc_cuda_executor_interruptible_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    cudaStream_t stream = NULL;
    ucc_ec_cuda_executor_interruptible_task_t *ee_task;
    ucc_status_t status;

    status = ucc_cuda_executor_interruptible_get_stream(&stream);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

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
                                           stream));
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start memcpy op");
            goto free_task;
        }
        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_COPY_MULTI:
        status = ucc_ec_cuda_copy_multi_kernel(task_args, stream);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start copy multi op");
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
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI:
        status = ucc_mc_reduce_multi(task_args->bufs[1], task_args->bufs[2],
                                     task_args->bufs[0], task_args->size,
                                     task_args->count, task_args->stride,
                                     task_args->dt, task_args->op,
                                     UCC_MEMORY_TYPE_CUDA);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start reduce multi op");
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
                                           UCC_MEMORY_TYPE_CUDA);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start reduce multi alpha op");
            goto free_task;
        }
        break;
    default:
        ec_error(&ucc_ec_cuda.super, "executor operation %d is not supported",
                 task_args->task_type);
        status = UCC_ERR_INVALID_PARAM;
        goto free_task;
    }

    status = ucc_ec_cuda_event_post(stream, ee_task->event);
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

ucc_status_t ucc_cuda_executor_interruptible_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);

    eee->state = UCC_EC_CUDA_EXECUTOR_INITIALIZED;
    return UCC_OK;
}
