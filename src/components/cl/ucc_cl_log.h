/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_LOG_H_
#define UCC_CL_LOG_H_
#include "utils/ucc_log.h"
#define ucc_log_component_cl(_cl_lib, _level, fmt, ...)                        \
    ucc_log_component(_level, ((ucc_base_lib_t *)_cl_lib)->log_component, fmt, \
                      ##__VA_ARGS__)

#define cl_error(_cl_lib, _fmt, ...)                                           \
    ucc_log_component_cl(_cl_lib, UCS_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define cl_warn(_cl_lib, _fmt, ...)                                            \
    ucc_log_component_cl(_cl_lib, UCS_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define cl_info(_cl_lib, _fmt, ...)                                            \
    ucc_log_component_cl(_cl_lib, UCS_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define cl_debug(_cl_lib, _fmt, ...)                                           \
    ucc_log_component_cl(_cl_lib, UCS_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define cl_trace(_cl_lib, _fmt, ...)                                           \
    ucc_log_component_cl(_cl_lib, UCS_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)

#endif
