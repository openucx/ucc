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

typedef struct ucc_team_lib* ucc_team_lib_h;

typedef struct ucc_team_lib_config {
    /* Team library priority */
    int                        priority;
} ucc_team_lib_config_t;

extern ucs_config_field_t ucc_team_lib_config_table[];

typedef struct ucc_team_lib {
    char*                          name;
    int                            priority;
    ucc_lib_params_t               params;
    void*                          dl_handle;
    ucs_config_global_list_entry_t team_lib_config;
    void                           (*cleanup)(ucc_team_lib_h self);
} ucc_team_lib_t;


#endif
