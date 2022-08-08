/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CPU_H_
#define UCC_EC_CPU_H_

#include "components/ec/base/ucc_ec_base.h"
#include "components/ec/ucc_ec_log.h"
#include "utils/ucc_mpool.h"

typedef struct ucc_ec_cpu_config {
    ucc_ec_config_t super;
} ucc_ec_cpu_config_t;

typedef struct ucc_ec_cpu {
    ucc_ec_base_t     super;
    ucc_thread_mode_t thread_mode;
    ucc_mpool_t       executors;
    ucc_mpool_t       executor_tasks;
    ucc_spinlock_t    init_spinlock;
} ucc_ec_cpu_t;

extern ucc_ec_cpu_t ucc_ec_cpu;

ucc_status_t ucc_ec_cpu_reduce(ucc_eee_task_reduce_t *task, uint16_t flags);
#endif
