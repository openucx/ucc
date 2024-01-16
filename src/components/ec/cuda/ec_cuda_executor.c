/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda_executor.h"

ucc_status_t ucc_cuda_executor_interruptible_start(ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_interruptible_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_persistent_start(ucc_ee_executor_t *executor,
                                                void *ee_context);

ucc_status_t ucc_cuda_executor_persistent_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_persistent_wait_start(ucc_ee_executor_t *executor,
                                                     void *ee_context);

ucc_status_t ucc_cuda_executor_persistent_wait_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_init(const ucc_ee_executor_params_t *params,
                                    ucc_ee_executor_t **executor)
{
    ucc_ec_cuda_executor_t  *eee;
    ucc_ec_cuda_resources_t *resources;
    ucc_status_t             status;

    status = ucc_ec_cuda_get_resources(&resources);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    eee = ucc_mpool_get(&resources->executors);
    if (ucc_unlikely(!eee)) {
        ec_error(&ucc_ec_cuda.super, "failed to allocate executor");
        return UCC_ERR_NO_MEMORY;
    }

    if (params->mask & UCC_EE_EXECUTOR_PARAM_FIELD_TASK_TYPES) {
        eee->requested_ops = params->task_types;
    } else {
        /* if no task types provided assume all tasks types required */
        eee->requested_ops = 1;
    }

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

ucc_status_t ucc_cuda_executor_start(ucc_ee_executor_t *executor,
                                     void *ee_context)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);

    if (!ee_context) {
        return ucc_cuda_executor_interruptible_start(executor);
    } else {
        if (eee->requested_ops == 0) {
            /* no operations requested, just mark stream busy */
            return ucc_cuda_executor_persistent_wait_start(executor,
                                                           ee_context);
        } else {
            return ucc_cuda_executor_persistent_start(executor, ee_context);
        }
    }
}

ucc_status_t ucc_cuda_executor_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    if (eee->mode == UCC_EC_CUDA_EXECUTOR_MODE_INTERRUPTIBLE) {
        return ucc_cuda_executor_interruptible_stop(executor);
    } else {
        if (eee->requested_ops == 0) {
            return ucc_cuda_executor_persistent_wait_stop(executor);
        } else {
            return ucc_cuda_executor_persistent_stop(executor);
        }
    }
}
