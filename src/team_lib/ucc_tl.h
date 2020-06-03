/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_TL_H_
#define UCC_TL_H_

#include "config.h"
#include "api/ucc.h"
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include <assert.h>
#include <string.h>

typedef struct ucc_team_lib ucc_team_lib_t;

typedef struct ucc_tl_lib_config {
    /* Log level above which log messages will be printed */
    ucs_log_component_config_t log_component;
    /* Team library priority */
    int                        priority;
} ucc_tl_lib_config_t;
extern ucs_config_field_t ucc_tl_lib_config_table[];

typedef struct ucc_tl_context_config {
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
} ucc_tl_iface_t;

typedef struct ucc_team_lib {
    ucc_tl_iface_t             *iface;
    ucs_log_component_config_t log_component;
    int                        priority;
} ucc_team_lib_t;

#endif
