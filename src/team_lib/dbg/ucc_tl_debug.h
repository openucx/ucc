/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_DEBUG_TEAM_H_
#define UCC_DEBUG_TEAM_H_
#include "team_lib/ucc_tl.h"

typedef struct ucc_tl_debug_iface {
    ucc_tl_iface_t super;
} ucc_tl_debug_iface_t;
extern ucc_tl_debug_iface_t ucc_team_lib_debug;

typedef struct ucc_tl_debug_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_debug_lib_config_t;

typedef struct ucc_tl_debug_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_debug_context_config_t;

typedef struct ucc_tl_debug {
    ucc_team_lib_t super;
    //ucs_log_component_config_t log_component;
} ucc_tl_debug_t;

#endif
