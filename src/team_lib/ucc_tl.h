/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_TL_H_
#define UCC_TL_H_

#include "config.h"
#include "api/ucc.h"
#include "core/ucc_lib.h"
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include <assert.h>
#include <string.h>

typedef struct ucc_team_lib   ucc_team_lib_t;
typedef struct ucc_tl_iface   ucc_tl_iface_t;
typedef struct ucc_tl_context ucc_tl_context_t;
typedef struct ucc_tl_team    ucc_tl_team_t;

typedef struct ucc_tl_lib_config {
    /* Log level above which log messages will be printed */
    ucs_log_component_config_t log_component;
    /* Team library priority */
    int                        priority;
} ucc_tl_lib_config_t;
extern ucs_config_field_t ucc_tl_lib_config_table[];

typedef struct ucc_tl_context_config {
    ucc_tl_iface_t *iface;
    ucc_team_lib_t *tl_lib;
} ucc_tl_context_config_t;
extern ucs_config_field_t ucc_tl_context_config_table[];

typedef struct ucc_tl_iface {
    char*                          name;
    int                            priority;
    ucc_lib_params_t               params;
    void*                          dl_handle;
    ucs_config_global_list_entry_t tl_lib_config;
    ucs_config_global_list_entry_t tl_context_config;
    ucc_status_t                   (*init)(const ucc_lib_params_t *params,
                                           const ucc_lib_config_t *config,
                                           const ucc_tl_lib_config_t *tl_config,
                                           ucc_team_lib_t **tl_lib);
    void                           (*cleanup)(ucc_team_lib_t *tl_lib);
    ucc_status_t                   (*context_create)(ucc_team_lib_t *tl_lib,
                                                     const ucc_context_params_t *params,
                                                     const ucc_tl_context_config_t *config,
                                                     ucc_tl_context_t **tl_context);
    void                           (*context_destroy)(ucc_tl_context_t *tl_context);
    ucc_status_t                   (*team_create_post)(ucc_tl_context_t **tl_ctxs,
                                                       uint32_t n_ctxs,
                                                       const ucc_team_params_t *params,
                                                       ucc_tl_team_t **team);
    ucc_status_t                   (*team_create_test)(ucc_tl_team_t *tneam_ctx);
    ucc_status_t                   (*team_destroy)(ucc_tl_team_t *team);
} ucc_tl_iface_t;

typedef struct ucc_team_lib {
    ucc_tl_iface_t             *iface;
    ucs_log_component_config_t log_component;
    int                        priority;
} ucc_team_lib_t;

typedef struct ucc_tl_context {
    ucc_team_lib_t *tl_lib;
} ucc_tl_context_t;

typedef struct ucc_tl_team {
    ucc_tl_iface_t  *iface;
    ucc_team_lib_t  *tl_lib;
} ucc_tl_team_t;

#define UCC_TL_LOG(_log_component, _level, _fmt, ...) do {             \
        ucs_log_component(_level, &_log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define ucc_tl_error(_lib, _fmt, ...)        \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define ucc_tl_warn(_lib, _fmt, ...)         \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define ucc_tl_info(_lib, _fmt, ...)         \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define ucc_tl_debug(_lib, _fmt, ...)        \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define ucc_tl_trace(_lib, _fmt, ...)        \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define ucc_tl_trace_req(_lib, _fmt, ...)    \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define ucc_tl_trace_data(_lib, _fmt, ...)   \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define ucc_tl_trace_async(_lib, _fmt, ...)  \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define ucc_tl_trace_func(_lib, _fmt, ...)   \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define ucc_tl_trace_poll(_lib, _fmt, ...)   \
    UCC_TL_LOG(_lib->log_component, UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

#endif
