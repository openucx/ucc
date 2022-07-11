/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_ROCM_H_
#define UCC_EC_ROCM_H_

#include "components/ec/base/ucc_ec_base.h"
#include "components/ec/ucc_ec_log.h"
#include "utils/ucc_mpool.h"
#include "utils/arch/rocm_def.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

typedef enum ucc_ec_rocm_strm_task_mode {
    UCC_EC_ROCM_TASK_KERNEL,
    UCC_EC_ROCM_TASK_MEM_OPS,
    UCC_EC_ROCM_TASK_AUTO,
    UCC_EC_ROCM_TASK_LAST,
} ucc_ec_rocm_strm_task_mode_t;

typedef enum ucc_ec_rocm_task_stream_type {
    UCC_EC_ROCM_USER_STREAM,
    UCC_EC_ROCM_INTERNAL_STREAM,
    UCC_EC_ROCM_TASK_STREAM_LAST
} ucc_ec_rocm_task_stream_type_t;

typedef enum ucc_ec_task_status {
    UCC_EC_ROCM_TASK_COMPLETED,
    UCC_EC_ROCM_TASK_POSTED,
    UCC_EC_ROCM_TASK_STARTED,
    UCC_EC_ROCM_TASK_COMPLETED_ACK
} ucc_ec_task_status_t;

typedef enum ucc_ec_rocm_executor_state {
    UCC_EC_ROCM_EXECUTOR_INITIALIZED,
    UCC_EC_ROCM_EXECUTOR_POSTED,
    UCC_EC_ROCM_EXECUTOR_STARTED,
    UCC_EC_ROCM_EXECUTOR_SHUTDOWN,
    UCC_EC_ROCM_EXECUTOR_SHUTDOWN_ACK
} ucc_ec_rocm_executor_state_t;

typedef enum ucc_ec_rocm_executor_mode {
    UCC_EC_ROCM_EXECUTOR_MODE_PERSISTENT,
    UCC_EC_ROCM_EXECUTOR_MODE_INTERRUPTIBLE
} ucc_ec_rocm_executor_mode_t;


static inline ucc_status_t hip_error_to_ucc_status(hipError_t hip_err)
{
    switch(hip_err) {
    case hipSuccess:
        return UCC_OK;
    case hipErrorNotReady:
        return UCC_INPROGRESS;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

typedef ucc_status_t (*ucc_ec_rocm_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  hipStream_t stream);

typedef struct ucc_ec_rocm_config {
    ucc_ec_config_t                super;
    ucc_ec_rocm_strm_task_mode_t   strm_task_mode;
    ucc_ec_rocm_task_stream_type_t task_strm_type;
    int                            stream_blocking_wait;
    unsigned long                  exec_num_workers;
    unsigned long                  exec_num_threads;
    unsigned long                  exec_max_tasks;
    unsigned long                  exec_num_streams;
} ucc_ec_rocm_config_t;

typedef struct ucc_ec_rocm {
    ucc_ec_base_t                  super;
    int                            stream_initialized;
    hipStream_t                    stream;
    int                            exec_streams_initialized;
    hipStream_t                   *exec_streams;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_mpool_t                    executors;
    ucc_mpool_t                    executor_interruptible_tasks;
    ucc_thread_mode_t              thread_mode;
    ucc_ec_rocm_strm_task_mode_t   strm_task_mode;
    ucc_ec_rocm_task_stream_type_t task_strm_type;
    ucc_ec_rocm_task_post_fn       post_strm_task;
    ucc_spinlock_t                 init_spinlock;
} ucc_ec_rocm_t;

typedef struct ucc_rocm_ec_event {
    hipEvent_t    event;
} ucc_ec_rocm_event_t;

typedef struct ucc_ec_rocm_stream_request {
    uint32_t            status;
    uint32_t           *dev_status;
    hipStream_t         stream;
} ucc_ec_rocm_stream_request_t;

typedef struct ucc_ec_rocm_executor_interruptible_task {
    ucc_ee_executor_task_t  super;
    void                   *event;
} ucc_ec_rocm_executor_interruptible_task_t;

typedef struct ucc_ec_rocm_executor_task_ops {
    ucc_status_t (*task_post)(ucc_ee_executor_t *executor,
                              const ucc_ee_executor_task_args_t *task_args,
                              ucc_ee_executor_task_t **task);
    ucc_status_t (*task_test)(const ucc_ee_executor_task_t *task);
    ucc_status_t (*task_finalize)(ucc_ee_executor_task_t *task);
} ucc_ec_rocm_executor_task_ops_t;

typedef struct ucc_ec_rocm_executor {
    ucc_ee_executor_t                super;
    ucc_ec_rocm_executor_mode_t      mode;
    ucc_ec_rocm_executor_task_ops_t  ops;
    ucc_spinlock_t                   tasks_lock;
    ucc_ec_rocm_executor_state_t     state;
    int                              pidx;
    ucc_ee_executor_task_t          *tasks;
    ucc_ec_rocm_executor_state_t    *dev_state;
    ucc_ee_executor_task_t          *dev_tasks;
    int                             *dev_pidx;
    int                             *dev_cidx;
} ucc_ec_rocm_executor_t;

ucc_status_t ucc_ec_rocm_event_create(void **event);

ucc_status_t ucc_ec_rocm_event_destroy(void *event);

ucc_status_t ucc_ec_rocm_event_post(void *ee_context, void *event);

ucc_status_t ucc_ec_rocm_event_test(void *event);

extern ucc_ec_rocm_t ucc_ec_rocm;

#define EC_ROCM_CONFIG                                                         \
    (ucc_derived_of(ucc_ec_rocm.super.config, ucc_ec_rocm_config_t))

#define UCC_EC_ROCM_INIT_STREAM() do {                                         \
    if (!ucc_ec_rocm.stream_initialized) {                                     \
        hipError_t hip_st = hipSuccess;                                        \
        ucc_spin_lock(&ucc_ec_rocm.init_spinlock);                             \
        if (!ucc_ec_rocm.stream_initialized) {                                 \
            hip_st = hipStreamCreateWithFlags(&ucc_ec_rocm.stream,             \
                                                hipStreamNonBlocking);         \
            ucc_ec_rocm.stream_initialized = 1;                                \
        }								       \
        ucc_spin_unlock(&ucc_ec_rocm.init_spinlock);                           \
        if(hip_st != hipSuccess) {                                             \
            ec_error(&ucc_ec_rocm.super, "rocm failed with ret:%d(%s)",        \
                     hip_st, hipGetErrorString(hip_st));                       \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
    }                                                                          \
} while(0)

#endif
