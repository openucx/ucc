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
    ucc_memory_cpu_store_fence();
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
