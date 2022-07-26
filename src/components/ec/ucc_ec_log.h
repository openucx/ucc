/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_LOG_H_
#define UCC_EC_LOG_H_

#include "utils/ucc_log.h"

#define ucc_log_component_ec(_ec, _level, fmt, ...)                      \
    ucc_log_component(_level, ((_ec)->config)->log_component,            \
                      fmt, ##__VA_ARGS__)

#define ec_error(_ec, _fmt, ...)                                         \
    ucc_log_component_ec(_ec, UCC_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define ec_warn(_ec, _fmt, ...)                                          \
    ucc_log_component_ec(_ec, UCC_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define ec_info(_ec, _fmt, ...)                                          \
    ucc_log_component_ec(_ec, UCC_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define ec_debug(_ec, _fmt, ...)                                         \
    ucc_log_component_ec(_ec, UCC_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define ec_trace(_ec, _fmt, ...)                                         \
    ucc_log_component_ec(_ec, UCC_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)

#endif
