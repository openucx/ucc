/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_H_
#define UCC_EC_CUDA_H_

#include "components/ec/base/ucc_ec_base.h"
#include "components/ec/ucc_ec_log.h"
#include "utils/arch/cuda_def.h"
#include "utils/ucc_mpool.h"
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_SUBTASKS 12

typedef enum ucc_ec_cuda_strm_task_mode {
    UCC_EC_CUDA_TASK_KERNEL,
    UCC_EC_CUDA_TASK_MEM_OPS,
    UCC_EC_CUDA_TASK_AUTO,
    UCC_EC_CUDA_TASK_LAST,
} ucc_ec_cuda_strm_task_mode_t;

typedef enum ucc_ec_cuda_executor_state {
    UCC_EC_CUDA_EXECUTOR_INITIALIZED,
    UCC_EC_CUDA_EXECUTOR_POSTED,
    UCC_EC_CUDA_EXECUTOR_STARTED,
    UCC_EC_CUDA_EXECUTOR_SHUTDOWN,
    UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK
} ucc_ec_cuda_executor_state_t;

typedef enum ucc_ec_cuda_executor_mode {
    UCC_EC_CUDA_EXECUTOR_MODE_PERSISTENT,
    UCC_EC_CUDA_EXECUTOR_MODE_INTERRUPTIBLE
} ucc_ec_cuda_executor_mode_t;

typedef ucc_status_t (*ucc_ec_cuda_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  cudaStream_t stream);

typedef struct ucc_ec_cuda_config {
    ucc_ec_config_t                super;
    ucc_ec_cuda_strm_task_mode_t   strm_task_mode;
    unsigned long                  exec_num_workers;
    unsigned long                  exec_num_threads;
    unsigned long                  exec_max_tasks;
    unsigned long                  exec_num_streams;
    unsigned long                  reduce_num_blocks;
    int                            reduce_num_threads;
    int                            use_cooperative_launch;
    unsigned long                  exec_copy_thresh;
} ucc_ec_cuda_config_t;

typedef struct ucc_ec_cuda {
    ucc_ec_base_t                  super;
    int                            stream_initialized;
    cudaStream_t                   stream;
    int                            exec_streams_initialized;
    cudaStream_t                  *exec_streams;
    ucc_mpool_t                    events;
    ucc_mpool_t                    executors;
    ucc_mpool_t                    executor_interruptible_tasks;
    ucc_mpool_t                    executor_persistent_tasks;
    ucc_thread_mode_t              thread_mode;
    ucc_ec_cuda_strm_task_mode_t   strm_task_mode;
    ucc_spinlock_t                 init_spinlock;
} ucc_ec_cuda_t;

typedef struct ucc_ec_cuda_event {
    cudaEvent_t    event;
} ucc_ec_cuda_event_t;

typedef struct ucc_ec_cuda_stream_request {
    uint32_t            status;
    uint32_t           *dev_status;
    cudaStream_t        stream;
} ucc_ec_cuda_stream_request_t;

typedef struct ucc_ec_cuda_executor_interruptible_task {
    ucc_ee_executor_task_t  super;
    void                   *event;
    cudaGraph_t             graph;
    cudaGraphExec_t         graph_exec;
} ucc_ec_cuda_executor_interruptible_task_t;

typedef struct ucc_ec_cuda_executor_persistent_task {
    ucc_ee_executor_task_t       super;
    int                          num_subtasks;
    ucc_ee_executor_task_args_t *subtasks[MAX_SUBTASKS];
} ucc_ec_cuda_executor_persistent_task_t;

typedef struct ucc_ec_cuda_executor_task_ops {
    ucc_status_t (*task_post)(ucc_ee_executor_t *executor,
                              const ucc_ee_executor_task_args_t *task_args,
                              ucc_ee_executor_task_t **task);
    ucc_status_t (*task_test)(const ucc_ee_executor_task_t *task);
    ucc_status_t (*task_finalize)(ucc_ee_executor_task_t *task);
} ucc_ec_cuda_executor_task_ops_t;

typedef struct ucc_ec_cuda_executor {
    ucc_ee_executor_t                super;
    ucc_ec_cuda_executor_mode_t      mode;
    uint64_t                         requested_ops;
    ucc_ec_cuda_executor_task_ops_t  ops;
    ucc_spinlock_t                   tasks_lock;
    ucc_ec_cuda_executor_state_t     state;
    int                              pidx;
    ucc_ee_executor_task_args_t     *tasks;
    ucc_ec_cuda_executor_state_t    *dev_state;
    ucc_ee_executor_task_args_t     *dev_tasks;
    int                             *dev_pidx;
    int                             *dev_cidx;
} ucc_ec_cuda_executor_t;

ucc_status_t ucc_ec_cuda_event_create(void **event);

ucc_status_t ucc_ec_cuda_event_destroy(void *event);

ucc_status_t ucc_ec_cuda_event_post(void *ee_context, void *event);

ucc_status_t ucc_ec_cuda_event_test(void *event);

extern ucc_ec_cuda_t ucc_ec_cuda;

#define EC_CUDA_CONFIG                                                         \
    (ucc_derived_of(ucc_ec_cuda.super.config, ucc_ec_cuda_config_t))

#define UCC_EC_CUDA_INIT_STREAM() do {                                         \
    if (!ucc_ec_cuda.stream_initialized) {                                     \
        cudaError_t cuda_st = cudaSuccess;                                     \
        ucc_spin_lock(&ucc_ec_cuda.init_spinlock);                             \
        if (!ucc_ec_cuda.stream_initialized) {                                 \
            cuda_st = cudaStreamCreateWithFlags(&ucc_ec_cuda.stream,           \
                                                cudaStreamNonBlocking);        \
            ucc_ec_cuda.stream_initialized = 1;                                \
        }                                                                      \
        ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);                           \
        if (ucc_unlikely(cudaSuccess != cuda_st)) {                            \
            return cuda_error_to_ucc_status(cuda_st);                          \
        }                                                                      \
    }                                                                          \
} while(0)

#endif
