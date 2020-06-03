/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_BASIC_TEAM_H_
#define UCC_BASIC_TEAM_H_
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
    //ucs_log_component_config_t log_component;
} ucc_tl_basic_t;

#if 0
#define UCC_BASIC_LOG(_log_component, _level, _fmt, ...) do {             \
        ucs_log_component(_level, &_log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define ucc_basic_error(_lib, _fmt, ...)        \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define ucc_basic_warn(_lib, _fmt, ...)         \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define ucc_basic_info(_lib, _fmt, ...)         \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define ucc_basic_debug(_lib, _fmt, ...)        \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define ucc_basic_trace(_lib, _fmt, ...)        \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define ucc_basic_trace_req(_lib, _fmt, ...)    \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define ucc_basic_trace_data(_lib, _fmt, ...)   \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define ucc_basic_trace_async(_lib, _fmt, ...)  \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define ucc_basic_trace_func(_lib, _fmt, ...)   \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define ucc_basic_trace_poll(_lib, _fmt, ...)   \
    UCC_BASIC_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)
#endif
#endif
