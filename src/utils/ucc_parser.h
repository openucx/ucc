/**
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
typedef ucs_config_names_array_t ucc_config_names_array_t;

#define UCC_CONFIG_TYPE_LOG_COMP UCS_CONFIG_TYPE_LOG_COMP
#define UCC_CONFIG_REGISTER_TABLE UCS_CONFIG_REGISTER_TABLE
#define UCC_CONFIG_TYPE_STRING UCS_CONFIG_TYPE_STRING
#define UCC_CONFIG_TYPE_STRING_ARRAY UCS_CONFIG_TYPE_STRING_ARRAY

static inline ucc_status_t
ucc_config_parser_fill_opts(void *opts, ucc_config_field_t *fields,
                            const char *env_prefix, const char *table_prefix,
                            int ignore_errors)
{
    ucs_status_t status = ucs_config_parser_fill_opts(
        opts, fields, env_prefix, table_prefix, ignore_errors);
    return ucs_status_to_ucc_status(status);
}

static inline void
ucc_config_parser_release_opts(void *opts, ucc_config_field_t *fields)
{
    ucs_config_parser_release_opts(opts, fields);
}

static inline ucc_status_t
ucc_config_parser_set_value(void *opts, ucc_config_field_t *fields,
                            const char *name, const char *value)
{
    ucs_status_t status =
        ucs_config_parser_set_value(opts, fields, name, value);
    return ucs_status_to_ucc_status(status);
}

static inline void ucc_config_parser_print_opts(FILE *stream, const char *title,
                                                const void *opts,
                                                ucc_config_field_t *fields,
                                                const char *table_prefix,
                                                const char *prefix,
                                                ucc_config_print_flags_t flags)
{
    ucs_config_print_flags_t ucs_flags = 0;
    if (flags & UCC_CONFIG_PRINT_CONFIG) {
        ucs_flags |= UCS_CONFIG_PRINT_CONFIG;
    }
    if (flags & UCC_CONFIG_PRINT_HEADER) {
        ucs_flags |= UCS_CONFIG_PRINT_HEADER;
    }
    if (flags & UCC_CONFIG_PRINT_DOC) {
        ucs_flags |= UCS_CONFIG_PRINT_DOC;
    }
    if (flags & UCC_CONFIG_PRINT_HIDDEN) {
        ucs_flags |= UCS_CONFIG_PRINT_HIDDEN;
    }
    ucs_config_parser_print_opts(stream, title, opts, fields, table_prefix,
                                 prefix, ucs_flags);
}
#endif
