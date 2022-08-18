/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_ROCM_EXECUTOR_H_
#define UCC_EC_ROCM_EXECUTOR_H_

#include "ec_rocm.h"

ucc_status_t ucc_rocm_executor_init(const ucc_ee_executor_params_t *params,
                                    ucc_ee_executor_t **executor);

ucc_status_t ucc_rocm_executor_status(const ucc_ee_executor_t *executor);

ucc_status_t ucc_rocm_executor_finalize(ucc_ee_executor_t *executor);

ucc_status_t ucc_rocm_executor_start(ucc_ee_executor_t *executor,
                                     void *ee_context);

ucc_status_t ucc_rocm_executor_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_rocm_executor_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task);

ucc_status_t ucc_rocm_executor_task_test(const ucc_ee_executor_task_t *task);

ucc_status_t ucc_rocm_executor_task_finalize(ucc_ee_executor_task_t *task);

/* implemented in ec_rocm_executor.cu */
ucc_status_t ucc_ec_rocm_persistent_kernel_start(ucc_ec_rocm_executor_t *eee);

ucc_status_t ucc_ec_rocm_reduce(ucc_ee_executor_task_args_t *task,
                                hipStream_t                  stream);
#endif
