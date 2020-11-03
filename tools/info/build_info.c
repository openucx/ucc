/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_info.h"

void print_version()
{
    printf("# UCC version=%s revision %s\n", UCC_VERSION_STRING,
           UCC_GIT_REVISION);
}
