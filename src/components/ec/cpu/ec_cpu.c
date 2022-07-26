/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cpu.h"
#include "utils/arch/cpu.h"
#include "components/mc/ucc_mc.h"
#include <limits.h>

static ucc_config_field_t ucc_ec_cpu_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_ec_cpu_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_ec_config_table)},

    {NULL}

};

static ucc_status_t ucc_ec_cpu_init(const ucc_ec_params_t *ec_params)
{
    ucc_status_t status;

    ucc_strncpy_safe(ucc_ec_cpu.super.config->log_component.name,
                     ucc_ec_cpu.super.super.name,
                     sizeof(ucc_ec_cpu.super.config->log_component.name));
    ucc_ec_cpu.thread_mode = ec_params->thread_mode;

    status = ucc_mpool_init(&ucc_ec_cpu.executors, 0, sizeof(ucc_ee_executor_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX, NULL,
                            ec_params->thread_mode, "ec cpu executors");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cpu.super, "failed to created ec cpu executors mpool");
        return status;
    }

    status = ucc_mpool_init(&ucc_ec_cpu.executor_tasks, 0,
                            sizeof(ucc_ee_executor_task_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX, NULL,
                            ec_params->thread_mode, "ec cpu executor tasks");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cpu.super,
                 "failed to created ec cpu executor tasks mpool");
        ucc_mpool_cleanup(&ucc_ec_cpu.executors, 1);
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_ec_cpu_get_attr(ucc_ec_attr_t *ec_attr)
{
    if (ec_attr->field_mask & UCC_EC_ATTR_FIELD_THREAD_MODE) {
        ec_attr->thread_mode = ucc_ec_cpu.thread_mode;
    }

    return UCC_OK;
}

static ucc_status_t ucc_ec_cpu_finalize()
{
    ucc_mpool_cleanup(&ucc_ec_cpu.executors, 1);
    ucc_mpool_cleanup(&ucc_ec_cpu.executor_tasks, 1);

    return UCC_OK;
}

ucc_status_t ucc_cpu_executor_init(const ucc_ee_executor_params_t *params,
                                   ucc_ee_executor_t **executor)
{
    ucc_ee_executor_t *eee = ucc_mpool_get(&ucc_ec_cpu.executors);

    ec_debug(&ucc_ec_cpu.super, "executor init, eee: %p", eee);
    if (ucc_unlikely(!eee)) {
        ec_error(&ucc_ec_cpu.super, "failed to allocate executor");
        return UCC_ERR_NO_MEMORY;
    }

    eee->ee_type = params->ee_type;
    *executor = eee;

    return UCC_OK;
}

ucc_status_t ucc_cpu_executor_start(ucc_ee_executor_t *executor, //NOLINT
                                    void *ee_context)            //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_cpu_executor_status(const ucc_ee_executor_t *executor) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_cpu_executor_stop(ucc_ee_executor_t *executor) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_cpu_executor_task_post(ucc_ee_executor_t *executor,
                                        const ucc_ee_executor_task_args_t *task_args,
                                        ucc_ee_executor_task_t **task)
{
    ucc_ee_executor_task_t *eee_task;
    ucc_status_t            status;

    eee_task = ucc_mpool_get(&ucc_ec_cpu.executor_tasks);
    if (ucc_unlikely(!eee_task)) {
        return UCC_ERR_NO_MEMORY;
    }

    eee_task->eee = executor;
    switch (task_args->task_type) {
    case UCC_EE_EXECUTOR_TASK_TYPE_COPY:
        memcpy(task_args->bufs[0], task_args->bufs[1], task_args->count);
        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE:
        status = ucc_mc_reduce(task_args->bufs[1], task_args->bufs[2],
                               task_args->bufs[0], task_args->count,
                               task_args->dt, task_args->op,
                               UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cpu.super, "reduce op failed");
            goto free_task;
        }
        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI:
        status = ucc_mc_reduce_multi(task_args->bufs[1], task_args->bufs[2],
                                     task_args->bufs[0], task_args->size,
                                     task_args->count, task_args->stride,
                                     task_args->dt, task_args->op,
                                     UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cpu.super, "reduce multi op failed");
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
                                           UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            ec_error(&ucc_ec_cpu.super, "reduce multi alpha op failed");
            goto free_task;
        }
        break;
    case UCC_EE_EXECUTOR_TASK_TYPE_COPY_MULTI:
        status = UCC_ERR_NOT_SUPPORTED;
        goto free_task;
    }
    eee_task->status = UCC_OK;
    *task = eee_task;

    return UCC_OK;

free_task:
    ucc_mpool_put(eee_task);
    return status;
}

ucc_status_t ucc_cpu_executor_task_test(const ucc_ee_executor_task_t *task)
{
    return task->status;
}

ucc_status_t ucc_cpu_executor_task_finalize(ucc_ee_executor_task_t *task)
{
    ucc_mpool_put(task);
    return UCC_OK;
}

ucc_status_t ucc_cpu_executor_finalize(ucc_ee_executor_t *executor)
{
    ec_debug(&ucc_ec_cpu.super, "executor finalize, eee: %p", executor);
    ucc_mpool_put(executor);

    return UCC_OK;
}

ucc_ec_cpu_t ucc_ec_cpu = {
    .super.super.name                 = "cpu ec",
    .super.ref_cnt                    = 0,
    .super.type                       = UCC_EE_CPU_THREAD,
    .super.init                       = ucc_ec_cpu_init,
    .super.get_attr                   = ucc_ec_cpu_get_attr,
    .super.finalize                   = ucc_ec_cpu_finalize,
    .super.config_table =
        {
            .name   = "CPU execution component",
            .prefix = "EC_CPU_",
            .table  = ucc_ec_cpu_config_table,
            .size   = sizeof(ucc_ec_cpu_config_t),
        },
    .super.ops.task_post              = NULL,
    .super.ops.task_query             = NULL,
    .super.ops.task_end               = NULL,
    .super.ops.create_event           = NULL,
    .super.ops.destroy_event          = NULL,
    .super.ops.event_post             = NULL,
    .super.ops.event_test             = NULL,
    .super.executor_ops.init          = ucc_cpu_executor_init,
    .super.executor_ops.start         = ucc_cpu_executor_start,
    .super.executor_ops.status        = ucc_cpu_executor_status,
    .super.executor_ops.stop          = ucc_cpu_executor_stop,
    .super.executor_ops.task_post     = ucc_cpu_executor_task_post,
    .super.executor_ops.task_test     = ucc_cpu_executor_task_test,
    .super.executor_ops.task_finalize = ucc_cpu_executor_task_finalize,
    .super.executor_ops.finalize      = ucc_cpu_executor_finalize,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_ec_cpu.super.config_table,
                                &ucc_config_global_list);
