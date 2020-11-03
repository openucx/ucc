/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_INFO_H
#define UCC_INFO_H

#include "api/ucc.h"

enum {
    PRINT_VERSION = UCC_BIT(0),
};

void print_version();

#endif
