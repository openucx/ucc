/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_TL_BASIC_H_
#define UCC_TL_BASIC_H_
#include "team_lib/ucc_tl.h"

typedef struct ucc_tl_basic_iface {
    ucc_tl_iface_t super;
} ucc_tl_basic_iface_t;
extern ucc_tl_basic_iface_t ucc_team_lib_basic;

typedef struct ucc_tl_basic_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_basic_lib_config_t;

typedef struct ucc_tl_basic_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_basic_context_config_t;

typedef struct ucc_tl_basic {
    ucc_team_lib_t super;
} ucc_tl_basic_t;

#endif
