/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_UTILS_LOG_H_
#define UCC_UTILS_LOG_H_

#include "config.h"
#include "ucc_global_opts.h"

extern ucc_config_t ucc_lib_global_config;
#define ucc_log_component(_level, _fmt, ...)                            \
    do {                                                                \
        ucs_log_component(_level, &ucc_lib_global_config.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define ucc_error(_fmt, ...)        ucc_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define ucc_warn(_fmt, ...)         ucc_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define ucc_info(_fmt, ...)         ucc_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define ucc_debug(_fmt, ...)        ucc_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define ucc_trace(_fmt, ...)        ucc_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define ucc_trace_req(_fmt, ...)    ucc_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define ucc_trace_data(_fmt, ...)   ucc_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define ucc_trace_async(_fmt, ...)  ucc_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define ucc_trace_func(_fmt, ...)   ucc_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")",\
                                                      __FUNCTION__, ## __VA_ARGS__)
#define ucc_trace_poll(_fmt, ...)   ucc_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

#endif
