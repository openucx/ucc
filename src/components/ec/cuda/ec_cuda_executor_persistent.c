/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda_executor.h"
#include "utils/arch/cpu.h"

ucc_status_t
ucc_cuda_executor_persistent_task_post(ucc_ee_executor_t *executor,
                                       const ucc_ee_executor_task_args_t *task_args,
                                       ucc_ee_executor_task_t **task)
{
    ucc_ec_cuda_executor_t *eee       = ucc_derived_of(executor,
                                                       ucc_ec_cuda_executor_t);
    int                     max_tasks = EC_CUDA_CONFIG->exec_max_tasks;
    ucc_ee_executor_task_args_t            *subtask_args;
    ucc_ec_cuda_executor_persistent_task_t *ee_task;
    int                                     i;
    ucc_ec_cuda_resources_t                *resources;
    ucc_status_t                            status;

    status = ucc_ec_cuda_get_resources(&resources);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    if (ucc_ec_cuda.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spin_lock(&eee->tasks_lock);
    }

    ee_task = ucc_mpool_get(&resources->executor_persistent_tasks);
    if (ucc_unlikely(!ee_task)) {
        return UCC_ERR_NO_MEMORY;
    }

    ee_task->super.status = UCC_INPROGRESS;
    ee_task->super.eee    = executor;

    if (task_args->task_type == UCC_EE_EXECUTOR_TASK_COPY_MULTI) {
        ee_task->num_subtasks = task_args->copy_multi.num_vectors;
        for (i = 0; i < ee_task->num_subtasks; i++) {
            subtask_args = &(eee->tasks[(eee->pidx + i) % max_tasks]);
            subtask_args->task_type = UCC_EE_EXECUTOR_TASK_COPY;
            subtask_args->copy.src  = task_args->copy_multi.src[i];
            subtask_args->copy.dst  = task_args->copy_multi.dst[i];
            subtask_args->copy.len  = task_args->copy_multi.counts[i];

            ee_task->subtasks[i] = subtask_args;
        }
    } else if (task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_MULTI_DST) {
        ee_task->num_subtasks = task_args->reduce_multi_dst.n_bufs;
        for (i = 0; i < ee_task->num_subtasks; i++) {
            subtask_args = &(eee->tasks[(eee->pidx + i) % max_tasks]);
            subtask_args->task_type      = UCC_EE_EXECUTOR_TASK_REDUCE;
            subtask_args->reduce.srcs[0] = task_args->reduce_multi_dst.src1[i];
            subtask_args->reduce.srcs[1] = task_args->reduce_multi_dst.src2[i];
            subtask_args->reduce.dst     = task_args->reduce_multi_dst.dst[i];
            subtask_args->reduce.count   = task_args->reduce_multi_dst.counts[i];
            subtask_args->reduce.dt      = task_args->reduce_multi_dst.dt;
            subtask_args->reduce.op      = task_args->reduce_multi_dst.op;
            subtask_args->reduce.n_srcs  = 2;

            ee_task->subtasks[i] = subtask_args;
        }
    } else {
        ee_task->num_subtasks = 1;
        ee_task->subtasks[0] = &(eee->tasks[eee->pidx % max_tasks]);
        memcpy(ee_task->subtasks[0], task_args,
               sizeof(ucc_ee_executor_task_args_t));
    }
    ucc_memory_cpu_store_fence();
    eee->pidx += ee_task->num_subtasks;
    if (ucc_ec_cuda.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spin_unlock(&eee->tasks_lock);
    }
    ec_debug(&ucc_ec_cuda.super, "executor task post, eee: %p", eee);

    *task = &ee_task->super;
    return UCC_OK;
}

ucc_status_t
ucc_cuda_executor_persistent_task_test(const ucc_ee_executor_task_t *task)
{
    ucc_ec_cuda_executor_persistent_task_t *ee_task;
    ucc_status_t status;
    int i;

    ee_task = ucc_derived_of(task, ucc_ec_cuda_executor_persistent_task_t);

    if (ee_task->super.status != UCC_INPROGRESS) {
        goto exit;
    }

    status = CUDA_FUNC(cudaGetLastError());
    if (ucc_unlikely(status != UCC_OK)) {
        ee_task->super.status = status;
        goto exit;
    }

    for (i = 0; i < ee_task->num_subtasks; i++) {
        if (ee_task->subtasks[i]->task_type != UCC_EE_EXECUTOR_TASK_LAST) {
            goto exit;
        }
    }
    ee_task->super.status = UCC_OK;
exit:
    return ee_task->super.status;
}

ucc_status_t
ucc_cuda_executor_persistent_task_finalize(ucc_ee_executor_task_t *task)
{
    ucc_assert(task->status == UCC_OK);
    ucc_mpool_put(task);
    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_persistent_start(ucc_ee_executor_t *executor,
                                                void *ee_context)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    ucc_ec_cuda_resources_t *resources;
    ucc_status_t status;

    ucc_assert(eee->state == UCC_EC_CUDA_EXECUTOR_INITIALIZED);
    ec_debug(&ucc_ec_cuda.super, "executor start, eee: %p", eee);
    eee->super.ee_context = ee_context;
    eee->state            = UCC_EC_CUDA_EXECUTOR_POSTED;
    eee->pidx             = 0;
    eee->mode             = UCC_EC_CUDA_EXECUTOR_MODE_PERSISTENT;

    status = ucc_ec_cuda_get_resources(&resources);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    status = ucc_ec_cuda_persistent_kernel_start(
        eee, resources->num_threads_exec, resources->num_blocks_exec);
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cuda.super, "failed to launch executor kernel");
        return status;
    }

    eee->ops.task_post     = ucc_cuda_executor_persistent_task_post;
    eee->ops.task_test     = ucc_cuda_executor_persistent_task_test;
    eee->ops.task_finalize = ucc_cuda_executor_persistent_task_finalize;
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
