/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_EXECUTOR_H_
#define UCC_EC_CUDA_EXECUTOR_H_

#include "ec_cuda.h"

ucc_status_t ucc_cuda_executor_init(const ucc_ee_executor_params_t *params,
                                    ucc_ee_executor_t **executor);

ucc_status_t ucc_cuda_executor_status(const ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_finalize(ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_start(ucc_ee_executor_t *executor,
                                     void *ee_context);

ucc_status_t ucc_cuda_executor_stop(ucc_ee_executor_t *executor);

ucc_status_t ucc_cuda_executor_task_post(ucc_ee_executor_t *executor,
                                         const ucc_ee_executor_task_args_t *task_args,
                                         ucc_ee_executor_task_t **task);

ucc_status_t ucc_cuda_executor_task_test(const ucc_ee_executor_task_t *task);

ucc_status_t ucc_cuda_executor_task_finalize(ucc_ee_executor_task_t *task);

/* implemented in ec_cuda_executor.cu */
ucc_status_t ucc_ec_cuda_persistent_kernel_start(ucc_ec_cuda_executor_t *eee);

ucc_status_t ucc_ec_cuda_reduce(ucc_ee_executor_task_args_t *task,
                                cudaStream_t                 stream);
#endif
