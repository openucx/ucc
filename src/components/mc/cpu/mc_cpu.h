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
} ucc_mc_cpu_config_t;

typedef struct ucc_mc_cpu {
    ucc_mc_base_t super;
} ucc_mc_cpu_t;

extern ucc_mc_cpu_t ucc_mc_cpu;
#endif
