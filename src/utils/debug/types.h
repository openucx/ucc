/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_LOG_TYPES_H_
#define UCC_LOG_TYPES_H_

/**
 * Logging levels.
 */
 typedef enum {
    UCC_LOG_LEVEL_FATAL,        /* Immediate termination */
    UCC_LOG_LEVEL_ERROR,        /* Error is returned to the user */
    UCC_LOG_LEVEL_WARN,         /* Something's wrong, but we continue */
    UCC_LOG_LEVEL_DIAG,         /* Diagnostics, silent adjustments or internal error handling */
    UCC_LOG_LEVEL_INFO,         /* Information */
    UCC_LOG_LEVEL_DEBUG,        /* Low-volume debugging */
    UCC_LOG_LEVEL_TRACE,        /* High-volume debugging */
    UCC_LOG_LEVEL_TRACE_REQ,    /* Every send/receive request */
    UCC_LOG_LEVEL_TRACE_DATA,   /* Data sent/received on the transport */
    UCC_LOG_LEVEL_TRACE_ASYNC,  /* Asynchronous progress engine */
    UCC_LOG_LEVEL_TRACE_FUNC,   /* Function calls */
    UCC_LOG_LEVEL_TRACE_POLL,   /* Polling functions */
    UCC_LOG_LEVEL_LAST,
    UCC_LOG_LEVEL_PRINT         /* Temporary output */
} ucc_log_level_t;

/**
 * Logging component.
 */
 typedef struct ucc_log_component_config {
    ucc_log_level_t log_level;
    char            name[16];
    const char      *file_filter; /* glob pattern of source files */
} ucc_log_component_config_t;

#endif
