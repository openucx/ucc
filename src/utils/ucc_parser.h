/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_PARSER_H_
#define UCC_PARSER_H_

#include "config.h"
#include "ucc/api/ucc_status.h"
#include "ucc/api/ucc_def.h"
#include "utils/ucc_datastruct.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_list.h"

#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <ucs/config/parser.h>
#include <ucs/config/ini.h>

typedef ucs_config_field_t             ucc_config_field_t;
typedef ucs_config_names_array_t       ucc_config_names_array_t;
typedef ucs_config_global_list_entry_t ucc_config_global_list_entry_t;
typedef ucs_config_allow_list_t        ucc_config_allow_list_t;

typedef struct ucc_file_config ucc_file_config_t;

#define UCC_CONFIG_TYPE_LOG_COMP        UCS_CONFIG_TYPE_LOG_COMP
#define UCC_CONFIG_REGISTER_TABLE       UCS_CONFIG_REGISTER_TABLE
#define UCC_CONFIG_REGISTER_TABLE_ENTRY UCS_CONFIG_REGISTER_TABLE_ENTRY
#define UCC_CONFIG_TYPE_STRING          UCS_CONFIG_TYPE_STRING
#define UCC_CONFIG_TYPE_INT             UCS_CONFIG_TYPE_INT
#define UCC_CONFIG_TYPE_UINT            UCS_CONFIG_TYPE_UINT
#define UCC_CONFIG_TYPE_STRING_ARRAY    UCS_CONFIG_TYPE_STRING_ARRAY
#define UCC_CONFIG_TYPE_ALLOW_LIST      UCS_CONFIG_TYPE_ALLOW_LIST
#define UCC_CONFIG_TYPE_ARRAY           UCS_CONFIG_TYPE_ARRAY
#define UCC_CONFIG_TYPE_TABLE           UCS_CONFIG_TYPE_TABLE
#define UCC_CONFIG_TYPE_ULUNITS         UCS_CONFIG_TYPE_ULUNITS
#define UCC_CONFIG_TYPE_ENUM            UCS_CONFIG_TYPE_ENUM
#define UCC_CONFIG_TYPE_MEMUNITS        UCS_CONFIG_TYPE_MEMUNITS
#define UCC_ULUNITS_AUTO                UCS_ULUNITS_AUTO
#define UCC_CONFIG_TYPE_BITMAP          UCS_CONFIG_TYPE_BITMAP
#define UCC_CONFIG_TYPE_MEMUNITS        UCS_CONFIG_TYPE_MEMUNITS
#define UCC_CONFIG_TYPE_BOOL            UCS_CONFIG_TYPE_BOOL
#define UCC_CONFIG_ALLOW_LIST_NEGATE    UCS_CONFIG_ALLOW_LIST_NEGATE
#define UCC_CONFIG_ALLOW_LIST_ALLOW_ALL UCS_CONFIG_ALLOW_LIST_ALLOW_ALL
#define UCC_CONFIG_ALLOW_LIST_ALLOW     UCS_CONFIG_ALLOW_LIST_ALLOW

/* Convenience structure used, for example, to represent TLS list.
   "requested" field is set to 1 if the list of entries was
   explicitly requested by user.

   Union with allow_list is used in order to use this structure
   directly as config_field. */
typedef struct ucc_config_names_list {
    union {
        struct {
            ucc_config_names_array_t array;
            int                      requested;
        };
        ucc_config_allow_list_t list;
    };
} ucc_config_names_list_t;

ucc_status_t ucc_config_parser_fill_opts(void *opts, ucc_config_field_t *fields,
                                         const char *env_prefix,
                                         const char *table_prefix,
                                         int         ignore_errors);

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

static inline ucs_config_print_flags_t
ucc_print_flags_to_ucs_print_flags(ucc_config_print_flags_t flags)
{
    int ucs_flags = 0;

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

    return (ucs_config_print_flags_t)ucs_flags;
}

static inline void ucc_config_parser_print_opts(FILE *stream, const char *title,
                                                const void *opts,
                                                ucc_config_field_t *fields,
                                                const char *table_prefix,
                                                const char *prefix,
                                                ucc_config_print_flags_t flags)
{
    ucs_config_print_flags_t ucs_flags;

    ucs_flags = ucc_print_flags_to_ucs_print_flags(flags);
    ucs_config_parser_print_opts(stream, title, opts, fields, table_prefix,
                                 prefix, ucs_flags);
}

void ucc_config_parser_print_all_opts(FILE *stream, const char *prefix,
                                      ucc_config_print_flags_t flags,
                                      ucc_list_link_t *        config_list);

ucc_status_t ucc_config_names_array_dup(ucc_config_names_array_t *dst,
                                        const ucc_config_names_array_t *src);

ucc_status_t ucc_config_names_array_merge(ucc_config_names_array_t *dst,
                                          const ucc_config_names_array_t *src);

void ucc_config_names_array_free(ucc_config_names_array_t *array);

int ucc_config_names_search(const ucc_config_names_array_t *config_names,
                            const char *                    str);

static inline
int ucc_config_names_array_is_all(const ucc_config_names_array_t *array)
{
    return (array->count == 1) && (0 == strcmp(array->names[0], "all"));
}

ucc_status_t ucc_config_allow_list_process(const ucc_config_allow_list_t * list,
                                           const ucc_config_names_array_t *all,
                                           ucc_config_names_list_t *       out);

ucc_status_t ucc_parse_file_config(const char *        filename,
                                   ucc_file_config_t **cfg);
void         ucc_release_file_config(ucc_file_config_t *cfg);

#endif
