/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_LOG_H_
#define UCC_MC_LOG_H_

#include "utils/ucc_log.h"

#define ucc_log_component_mc(_mc, _level, fmt, ...)                      \
    ucc_log_component(_level, ((_mc)->config)->log_component,            \
                      fmt, ##__VA_ARGS__)

#define mc_error(_mc, _fmt, ...)                                         \
    ucc_log_component_mc(_mc, UCC_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define mc_warn(_mc, _fmt, ...)                                          \
    ucc_log_component_mc(_mc, UCC_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define mc_info(_mc, _fmt, ...)                                          \
    ucc_log_component_mc(_mc, UCC_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define mc_debug(_mc, _fmt, ...)                                         \
    ucc_log_component_mc(_mc, UCC_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define mc_trace(_mc, _fmt, ...)                                         \
    ucc_log_component_mc(_mc, UCC_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)

#endif
