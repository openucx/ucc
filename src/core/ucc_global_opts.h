/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_GLOBAL_OPTS_H_
#define UCC_GLOBAL_OPTS_H_

#include "config.h"

#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>

typedef struct ucc_config {
    /* Log level above which log messages will be printed*/
    ucs_log_component_config_t log_component;

    /* Coll component libraries path */
    char                       *ccm_path;
} ucc_config_t;

extern ucc_config_t ucc_lib_global_config;
extern ucs_config_field_t ucc_lib_global_config_table[];

#endif
