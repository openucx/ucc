/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_SYSINFO_IB_H_
#define UCC_SYSINFO_IB_H_

#include "components/topo/base/ucc_sysinfo_base.h"

typedef struct ucc_sysinfo_ib {
    ucc_sysinfo_base_t super;
} ucc_sysinfo_ib_t;

extern ucc_sysinfo_ib_t ucc_sysinfo_ib;

#endif
