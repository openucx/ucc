/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda_executor.h"
#include "utils/ucc_atomic.h"

ucc_status_t ucc_cuda_executor_interruptible_get_stream(cudaStream_t *stream)
{
    static uint32_t          last_used   = 0;
    int                      num_streams = EC_CUDA_CONFIG->exec_num_streams;
    ucc_ec_cuda_resources_t *resources;
    ucc_status_t             st;
    int                      i, j;
    uint32_t                 id;

    ucc_assert(num_streams > 0);
    if (ucc_unlikely(num_streams <= 0)) {
        return UCC_ERR_INVALID_PARAM;
    }
    st = ucc_ec_cuda_get_resources(&resources);
    if (ucc_unlikely(st != UCC_OK)) {
        return st;
    }

    if (ucc_unlikely(!resources->streams_initialized)) {
        ucc_spin_lock(&ucc_ec_cuda.init_spinlock);
        if (resources->streams_initialized) {
            goto unlock;
        }

        for(i = 0; i < num_streams; i++) {
            st = CUDA_FUNC(cudaStreamCreateWithFlags(&resources->exec_streams[i],
                                                     cudaStreamNonBlocking));
            if (st != UCC_OK) {
                for (j = 0; j < i; j++) {
                    CUDA_FUNC(cudaStreamDestroy(resources->exec_streams[j]));
                }
                ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
                return st;
            }
        }
        resources->streams_initialized = 1;
unlock:
        ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
    }

    id = ucc_atomic_fadd32(&last_used, 1);
    *stream = resources->exec_streams[id % num_streams];
    return UCC_OK;
}


ucc_status_t ucc_ec_cuda_copy_multi_kernel(const ucc_ee_executor_task_args_t *args,
                                           cudaStream_t stream);

ucc_status_t
ucc_cuda_executor_interruptible_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    cudaStream_t stream    = NULL;
    size_t       num_nodes = UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS;
    ucc_ec_cuda_executor_interruptible_task_t *ee_task;
    ucc_status_t status;
    cudaGraphNode_t nodes[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    ucc_ec_cuda_resources_t *resources;
    int i;

    status = ucc_ec_cuda_get_resources(&resources);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_cuda_executor_interruptible_get_stream(&stream);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    ee_task = ucc_mpool_get(&resources->executor_interruptible_tasks);
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
    case UCC_EE_EXECUTOR_TASK_COPY:
        status = CUDA_FUNC(
            cudaMemcpyAsync(task_args->copy.dst, task_args->copy.src,
                            task_args->copy.len, cudaMemcpyDefault, stream));
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start memcpy op");
            goto free_task;
        }
        break;
    case UCC_EE_EXECUTOR_TASK_COPY_MULTI:
        if ((task_args->copy_multi.counts[0] > EC_CUDA_CONFIG->exec_copy_thresh) &&
            (task_args->copy_multi.num_vectors > 2)) {
            status = CUDA_FUNC(cudaGraphGetNodes(ee_task->graph, nodes,
                                                 &num_nodes));
            if (ucc_unlikely(status != UCC_OK)) {
                ec_error(&ucc_ec_cuda.super, "failed to get graph nodes");
                goto free_task;
            }
            for (i = 0; i < task_args->copy_multi.num_vectors; i++) {
                status = CUDA_FUNC(
                    cudaGraphExecMemcpyNodeSetParams1D(ee_task->graph_exec, nodes[i],
                                                       task_args->copy_multi.dst[i],
                                                       task_args->copy_multi.src[i],
                                                       task_args->copy_multi.counts[i],
                                                       cudaMemcpyDefault));
                if (ucc_unlikely(status != UCC_OK)) {
                    ec_error(&ucc_ec_cuda.super, "failed to instantiate graph");
                    goto free_task;
                }

            }
            for (; i < UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS; i++) {
                status = CUDA_FUNC(
                    cudaGraphExecMemcpyNodeSetParams1D(ee_task->graph_exec, nodes[i],
                                                       task_args->copy_multi.dst[0],
                                                       task_args->copy_multi.src[0],
                                                       1, cudaMemcpyDefault));
                if (ucc_unlikely(status != UCC_OK)) {
                    ec_error(&ucc_ec_cuda.super, "failed to instantiate graph");
                    goto free_task;
                }
            }

            status = CUDA_FUNC(cudaGraphLaunch(ee_task->graph_exec, stream));
            if (ucc_unlikely(status != UCC_OK)) {
                ec_error(&ucc_ec_cuda.super, "failed to instantiate graph");
                goto free_task;
            }

        } else {
            status = ucc_ec_cuda_copy_multi_kernel(task_args, stream);
            if (ucc_unlikely(status != UCC_OK)) {
                ec_error(&ucc_ec_cuda.super, "failed to start copy multi op");
                goto free_task;
            }
        }
        break;
    case UCC_EE_EXECUTOR_TASK_REDUCE:
    case UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED:
    case UCC_EE_EXECUTOR_TASK_REDUCE_MULTI_DST:
        status = ucc_ec_cuda_reduce(
            (ucc_ee_executor_task_args_t *)task_args,
            resources->num_threads_reduce,
            resources->num_blocks_reduce,
            stream);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cuda.super, "failed to start reduce op");
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

    ucc_assert(task->status == UCC_OK);
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
