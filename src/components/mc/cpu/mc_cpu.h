/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CPU_H_
#define UCC_MC_CPU_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"

typedef struct ucc_mc_cpu_config {
    ucc_mc_config_t super;
    size_t cpu_elem_size;
    int cpu_max_elems;
} ucc_mc_cpu_config_t;

typedef struct ucc_mc_cpu {
    ucc_mc_base_t super;
    ucc_mpool_t mpool;
    int mpool_init_flag;
    ucc_spinlock_t mpool_init_spinlock;
} ucc_mc_cpu_t;

extern ucc_mc_cpu_t ucc_mc_cpu;
#define MC_CPU_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_cpu.super.config, ucc_mc_cpu_config_t))
#endif
