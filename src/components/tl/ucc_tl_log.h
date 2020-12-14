/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_LOG_H_
#define UCC_TL_LOG_H_
#include "utils/ucc_log.h"
#define ucc_log_component_tl(_tl_ctx, _level, fmt, ...)                        \
    ucc_log_component(_level, ((ucc_base_context_t *)_tl_ctx)->lib->log_component, fmt, \
                      ##__VA_ARGS__)

#define tl_error(_tl_ctx, _fmt, ...)                                           \
    ucc_log_component_tl(_tl_ctx, UCC_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define tl_warn(_tl_ctx, _fmt, ...)                                            \
    ucc_log_component_tl(_tl_ctx, UCC_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define tl_info(_tl_ctx, _fmt, ...)                                            \
    ucc_log_component_tl(_tl_ctx, UCC_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define tl_debug(_tl_ctx, _fmt, ...)                                           \
    ucc_log_component_tl(_tl_ctx, UCC_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define tl_trace(_tl_ctx, _fmt, ...)                                           \
    ucc_log_component_tl(_tl_ctx, UCC_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)

#endif
