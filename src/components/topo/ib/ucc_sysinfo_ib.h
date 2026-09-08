/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef UCC_SYSINFO_IB_H_
#define UCC_SYSINFO_IB_H_

#include "components/topo/base/ucc_sysinfo_base.h"

typedef struct ucc_sysinfo_ib {
    ucc_sysinfo_base_t super;
} ucc_sysinfo_ib_t;

extern ucc_sysinfo_ib_t ucc_sysinfo_ib;

#endif
