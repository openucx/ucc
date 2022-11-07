/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm_executor.h"
#include "components/ec/ucc_ec.h"

ucc_status_t ucc_rocm_executor_persistent_start(ucc_ee_executor_t *executor,
                                                void *ee_context);

ucc_status_t ucc_rocm_executor_persistent_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_rocm_executor_interruptible_start(ucc_ee_executor_t *executor);

ucc_status_t ucc_rocm_executor_interruptible_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_rocm_executor_init(const ucc_ee_executor_params_t *params,
                                    ucc_ee_executor_t **executor)
{
    ucc_ec_rocm_executor_t  *eee = ucc_mpool_get(&ucc_ec_rocm.executors);
    ucc_status_t             status;
    ucc_ee_executor_params_t cpu_params = {
        .mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE,
        .ee_type = UCC_EE_CPU_THREAD
    };

    if (ucc_unlikely(!eee)) {
        ec_error(&ucc_ec_rocm.super, "failed to allocate executor");
        return UCC_ERR_NO_MEMORY;
    }

    ec_debug(&ucc_ec_rocm.super, "executor init, eee: %p", eee);
    eee->super.ee_type = params->ee_type;
    eee->state         = UCC_EC_ROCM_EXECUTOR_INITIALIZED;

    status = ucc_ee_executor_init(&cpu_params, &ucc_ec_rocm.cpu_executor);
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super,
                  "Error initializing CPU executor from ROCm component");
    }

    *executor = &eee->super;
    return UCC_OK;
}

ucc_status_t ucc_rocm_executor_status(const ucc_ee_executor_t *executor)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_rocm_executor_t);

    switch (eee->state) {
    case UCC_EC_ROCM_EXECUTOR_INITIALIZED:
        return UCC_OPERATION_INITIALIZED;
    case UCC_EC_ROCM_EXECUTOR_POSTED:
        return UCC_INPROGRESS;
    case UCC_EC_ROCM_EXECUTOR_STARTED:
        return UCC_OK;
    default:
/* executor has been destroyed */
        return UCC_ERR_NO_RESOURCE;
    }
}

ucc_status_t ucc_rocm_executor_finalize(ucc_ee_executor_t *executor)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_rocm_executor_t);
    ucc_status_t            status;

    ec_debug(&ucc_ec_rocm.super, "executor free, eee: %p", eee);
    ucc_assert(eee->state == UCC_EC_ROCM_EXECUTOR_INITIALIZED);
    ucc_mpool_put(eee);

    status = ucc_ee_executor_finalize(ucc_ec_rocm.cpu_executor);
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super,
                  "Error finalizing CPU executor from ROCm component");
    }

    return status;
}

ucc_status_t ucc_rocm_executor_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_rocm_executor_t);
    return eee->ops.task_post(executor, task_args, task);
}


ucc_status_t ucc_rocm_executor_task_test(const ucc_ee_executor_task_t *task)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(task->eee,
                                                 ucc_ec_rocm_executor_t);
    return eee->ops.task_test(task);
}

ucc_status_t ucc_rocm_executor_task_finalize(ucc_ee_executor_task_t *task)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(task->eee,
                                                 ucc_ec_rocm_executor_t);
    return eee->ops.task_finalize(task);
}

ucc_status_t ucc_rocm_executor_start(ucc_ee_executor_t *executor,
                                     void *ee_context)
{
    ucc_status_t            status;

    status = ucc_ee_executor_start(ucc_ec_rocm.cpu_executor, ee_context);
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super,
                  "Error starting CPU executor from ROCm component");
       return status;
    }

  if (!ee_context) {
        return ucc_rocm_executor_interruptible_start(executor);
    } else {
        return ucc_rocm_executor_persistent_start(executor, ee_context);
    }
}

ucc_status_t ucc_rocm_executor_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_rocm_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_rocm_executor_t);
    ucc_status_t            status;

    status = ucc_ee_executor_stop(ucc_ec_rocm.cpu_executor);
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super,
                  "Error stopping CPU executor from ROCm component");
       return status;
    }

    if (eee->mode == UCC_EC_ROCM_EXECUTOR_MODE_INTERRUPTIBLE) {
        return ucc_rocm_executor_interruptible_stop(executor);
    } else {
        return ucc_rocm_executor_persistent_stop(executor);
    }
}
