/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_PARSER_H_
#define UCC_PARSER_H_

#include "config.h"
#include "api/ucc_status.h"
#include "ucc_compiler_def.h"

#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <ucs/config/parser.h>

typedef ucs_config_field_t ucc_config_field_t;
#define UCC_CONFIG_TYPE_LOG_COMP UCS_CONFIG_TYPE_LOG_COMP
#define UCC_CONFIG_REGISTER_TABLE UCS_CONFIG_REGISTER_TABLE
#define UCC_CONFIG_TYPE_STRING UCS_CONFIG_TYPE_STRING

static inline ucc_status_t
ucc_config_parser_fill_opts(void *opts, ucc_config_field_t *fields,
                            const char *env_prefix, const char *table_prefix,
                            int ignore_errors)
{
    ucs_status_t status = ucs_config_parser_fill_opts(
        opts, fields, env_prefix, table_prefix, ignore_errors);
    return ucs_status_to_ucc_status(status);
}

#endif
