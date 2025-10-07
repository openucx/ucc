/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALLV_H_
#define ALLTOALLV_H_

#include "../tl_cuda.h"
#include "../tl_cuda_coll.h"

ucc_status_t ucc_tl_cuda_alltoallv_ce_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_alltoallv_ce_start(ucc_coll_task_t *coll_task);

void ucc_tl_cuda_alltoallv_ce_progress(ucc_coll_task_t *coll_task);

ucc_status_t
ucc_tl_cuda_alltoallv_ce_triggered_post_setup(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_alltoallv_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *tl_team,
                                        ucc_coll_task_t     **task_p);

/**
 * @brief Post a copy operation using CUDA copy engine
 *
 * This function posts a copy operation to be executed directly by the CUDA copy engine.
 * The executor and task parameters are unused as the operation is handled by CUDA.
 * The stream parameter is used to specify which CUDA stream should execute the copy.
 *
 * @param dst Destination buffer for the copy operation
 * @param src Source buffer for the copy operation
 * @param len Length of data to copy in bytes
 * @param executor Unused - operation handled by CUDA copy engine
 * @param task Unused - operation handled by CUDA copy engine
 * @param stream CUDA stream to execute the copy operation
 * @return UCC_OK on success, error code otherwise
 */
ucc_status_t cuda_copy_post(void *dst, const void *src, size_t len,
                            ucc_ee_executor_t       *executor,
                            ucc_ee_executor_task_t **task, cudaStream_t stream);

/**
 * @brief Post a copy operation using the UCC executor
 *
 * This function posts a copy operation to be executed by the UCC executor.
 * The executor and task parameters are used to track the operation's progress.
 * The stream parameter is unused as the executor manages its own execution context.
 *
 * @param dst Destination buffer for the copy operation
 * @param src Source buffer for the copy operation
 * @param len Length of data to copy in bytes
 * @param executor UCC executor to handle the copy operation
 * @param task Pointer to store the executor task handle
 * @param stream Unused - executor manages its own execution context
 * @return UCC_OK on success, error code otherwise
 */
ucc_status_t ee_copy_post(void *dst, const void *src, size_t len,
                          ucc_ee_executor_t       *executor,
                          ucc_ee_executor_task_t **task, cudaStream_t stream);

ucc_status_t
ucc_tl_cuda_alltoallv_ce_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                        ucc_coll_task_t *coll_task);

ucc_status_t
ucc_tl_cuda_alltoallv_ce_setup_copy_engine(ucc_tl_cuda_task_t *task,
                                           ucc_tl_cuda_lib_t  *lib,
                                           const char         *init_func_name);

#endif
