/**
 * Copyright (c) 2001-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_INFO_H
#define UCC_INFO_H

#include "ucc/api/ucc.h"

enum {
    PRINT_VERSION      = UCC_BIT(0),
    PRINT_BUILD_CONFIG = UCC_BIT(1),
};

void print_version();

void print_build_config();

#endif
