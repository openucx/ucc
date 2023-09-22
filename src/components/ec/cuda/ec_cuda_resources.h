/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_RESOURCES_H_
#define UCC_EC_CUDA_RESOURCES_H_

#include "components/ec/base/ucc_ec_base.h"
#include "utils/arch/cuda_def.h"
#include "utils/ucc_mpool.h"

#define MAX_SUBTASKS 12

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

typedef struct ucc_ec_cuda_event {
    cudaEvent_t event;
} ucc_ec_cuda_event_t;

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

typedef struct ucc_ec_cuda_resources {
    ucc_mpool_t events;
    ucc_mpool_t executors;
    ucc_mpool_t executor_interruptible_tasks;
    ucc_mpool_t executor_persistent_tasks;
} ucc_ec_cuda_resources_t;

ucc_status_t ucc_ec_cuda_resources_init(ucc_ec_base_t *ec,
                                        ucc_ec_cuda_resources_t *resources);

void ucc_ec_cuda_resources_cleanup(ucc_ec_cuda_resources_t *resources);

KHASH_INIT(ucc_ec_cuda_resources_hash, unsigned long long, void*, 1, \
           kh_int64_hash_func, kh_int64_hash_equal);
#define ucc_ec_cuda_resources_hash_t khash_t(ucc_ec_cuda_resources_hash)

static inline
void* ec_cuda_resources_hash_get(ucc_ec_cuda_resources_hash_t *h,
                                 unsigned long long key)
{
    khiter_t  k;
    void     *value;

    k = kh_get(ucc_ec_cuda_resources_hash, h , key);
    if (k == kh_end(h)) {
        return NULL;
    }
    value = kh_value(h, k);
    return value;
}

static inline
void ec_cuda_resources_hash_put(ucc_ec_cuda_resources_hash_t *h,
                                unsigned long long key,
                                void *value)
{
    int ret;
    khiter_t k;
    k = kh_put(ucc_ec_cuda_resources_hash, h, key, &ret);
    kh_value(h, k) = value;
}


#endif
