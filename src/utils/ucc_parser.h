/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
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
#include "utils/arch/cpu.h"
#include "khash.h"
#include "components/topo/ucc_topo.h"

#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <ucs/config/parser.h>

typedef ucs_config_field_t             ucc_config_field_t;
typedef ucs_config_names_array_t       ucc_config_names_array_t;
typedef ucs_config_global_list_entry_t ucc_config_global_list_entry_t;
typedef ucs_config_allow_list_t        ucc_config_allow_list_t;

typedef struct ucc_file_config ucc_file_config_t;

#if UCS_HAVE_CONFIG_GLOBAL_LIST_ENTRY_FLAGS
#define UCC_CONFIG_DECLARE_TABLE(_table, _name, _prefix, _type)                \
    static ucc_config_global_list_entry_t _table##_config_entry = {            \
        .name   = _name,                                                       \
        .prefix = _prefix,                                                     \
        .table  = _table,                                                      \
        .size   = sizeof(_type),                                               \
        .list   = {NULL, NULL},                                                \
        .flags  = 0                                                            \
    };
#else
#define UCC_CONFIG_DECLARE_TABLE(_table, _name, _prefix, _type)                \
    static ucc_config_global_list_entry_t _table##_config_entry = {            \
        .name   = _name,                                                       \
        .prefix = _prefix,                                                     \
        .table  = _table,                                                      \
        .size   = sizeof(_type),                                               \
        .list   = {NULL, NULL},                                                \
    };
#endif

#ifdef UCS_HAVE_PARSER_PRINT_FILTER_ARG
#define UCS_CONFIG_PARSER_PRINT_OPTS(_stream, _title, _opts, _fields, _tprefix, _prefix, _flags) \
    ucs_config_parser_print_opts((_stream), (_title), (_opts), (_fields), (_tprefix), (_prefix), (_flags), NULL)
#else
#define UCS_CONFIG_PARSER_PRINT_OPTS(_stream, _title, _opts, _fields, _tprefix, _prefix, _flags) \
    ucs_config_parser_print_opts((_stream), (_title), (_opts), (_fields), (_tprefix), (_prefix), (_flags))
#endif

#define UCC_CONFIG_GET_TABLE(_table)    &_table##_config_entry
#define UCC_CONFIG_TYPE_LOG_COMP        UCS_CONFIG_TYPE_LOG_COMP
#define UCC_CONFIG_REGISTER_TABLE       UCS_CONFIG_REGISTER_TABLE
#define UCC_CONFIG_REGISTER_TABLE_ENTRY UCS_CONFIG_REGISTER_TABLE_ENTRY
#define UCC_CONFIG_TYPE_LOG_COMP        UCS_CONFIG_TYPE_LOG_COMP
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
#define UCC_CONFIG_TYPE_TERNARY         UCS_CONFIG_TYPE_TERNARY
#define UCC_CONFIG_TYPE_ON_OFF_AUTO     UCS_CONFIG_TYPE_ON_OFF_AUTO
#define UCC_UUNITS_AUTO                 ((unsigned)-2)

typedef enum ucc_ternary_auto_value {
    UCC_NO   = UCS_NO,
    UCC_YES  = UCS_YES,
    UCC_TRY  = UCS_TRY,
    UCC_AUTO = UCS_AUTO,
    UCC_TERNARY_LAST
} ucc_ternary_auto_value_t;

typedef enum ucc_on_off_auto_value {
    UCC_CONFIG_OFF  = UCS_CONFIG_OFF,
    UCC_CONFIG_ON   = UCS_CONFIG_ON,
    UCC_CONFIG_AUTO = UCS_CONFIG_AUTO,
    UCC_CONFIG_ON_OFF_LAST
} ucc_on_off_auto_value_t;

enum tuning_mask {
    UCC_TUNING_DESC_FIELD_VENDOR    = UCC_BIT(0),
    UCC_TUNING_DESC_FIELD_MODEL     = UCC_BIT(1),
    UCC_TUNING_DESC_FIELD_TEAM_SIZE = UCC_BIT(2),
    UCC_TUNING_DESC_FIELD_PPN       = UCC_BIT(3),
    UCC_TUNING_DESC_FIELD_NNODES    = UCC_BIT(4),
    UCC_TUNING_DESC_FIELD_SOCK      = UCC_BIT(5)
};

typedef struct ucc_section_desc {
    uint64_t         mask;
    ucc_cpu_vendor_t vendor;
    ucc_cpu_model_t  model;
    ucc_rank_t       min_team_size;
    ucc_rank_t       max_team_size;
    ucc_rank_t       min_ppn;
    ucc_rank_t       max_ppn;
    ucc_rank_t       min_sock;
    ucc_rank_t       max_sock;
    ucc_rank_t       min_nnodes;
    ucc_rank_t       max_nnodes;
} ucc_section_desc_t;

KHASH_MAP_INIT_STR(ucc_sec, char *);

typedef struct ucc_section_wrap {
    ucc_section_desc_t desc;
    khash_t(ucc_sec)   vals_h;
} ucc_section_wrap_t;

KHASH_MAP_INIT_STR(ucc_cfg_file, char *);
KHASH_MAP_INIT_STR(ucc_sections, ucc_section_wrap_t *);

typedef struct ucc_file_config {
    char                  *filename;
    khash_t(ucc_cfg_file)  vars;
    khash_t(ucc_sections)  sections;
} ucc_file_config_t;

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

ucc_status_t ucc_config_parser_fill_opts(void *opts,
                                         ucs_config_global_list_entry_t *entry,
                                         const char *env_prefix,
                                         int ignore_errors);

ucc_status_t ucc_add_team_sections(void                *team_cfg,
                                   ucc_config_field_t  *tl_field,
                                   ucc_topo_t          *team_topo,
                                   const char         **tuning_str,
                                   const char          *tune_key,
                                   const char          *env_prefix,
                                   const char          *component_prefix);

static inline void
ucc_config_parser_release_opts(void *opts, ucc_config_field_t *fields)
{
    ucs_config_parser_release_opts(opts, fields);
}

static inline ucc_status_t
ucc_config_parser_set_value(void *opts, ucc_config_field_t *fields,
                            const char *name, const char *value)
{
    ucs_status_t status;

#if UCS_HAVE_PARSER_SET_VALUE_TABLE_PREFIX
    status = ucs_config_parser_set_value(opts, fields, NULL, name, value);
#else
    status = ucs_config_parser_set_value(opts, fields, name, value);
#endif
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
    UCS_CONFIG_PARSER_PRINT_OPTS(stream, title, opts, fields, table_prefix,
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

typedef struct ucc_pipeline_params ucc_pipeline_params_t;

ucc_status_t ucc_config_clone_table(const void *src, void *dst,
                                    const void *arg);

int ucc_pipeline_params_is_auto(const ucc_pipeline_params_t *p);

int ucc_config_sscanf_pipeline_params(const char *buf, void *dest,
                                      const void *arg);

int ucc_config_sprintf_pipeline_params(char *buf, size_t max, const void *src,
                                       const void *arg);

ucs_status_t ucc_config_clone_pipeline_params(const void *src, void *dest,
                                              const void *arg);

void ucc_config_release_pipeline_params(void *ptr, const void *arg);

int ucc_config_sscanf_uint_ranged(const char *buf, void *dest, const void *arg);

int ucc_config_sprintf_uint_ranged(char *buf, size_t max, const void *src,
                                   const void *arg);

ucs_status_t ucc_config_clone_uint_ranged(const void *src, void *dest,
                                          const void *arg);

void ucc_config_release_uint_ranged(void *ptr, const void *arg);

#ifdef UCS_HAVE_PARSER_CONFIG_DOC
#define UCC_CONFIG_TYPE_UINT_RANGED                                            \
    {                                                                          \
        ucc_config_sscanf_uint_ranged, ucc_config_sprintf_uint_ranged,         \
            ucc_config_clone_uint_ranged, ucc_config_release_uint_ranged,      \
            ucs_config_help_generic, ucs_config_doc_nop,                       \
            "[<munit>-<munit>:[mtype]:value,"                                  \
            "<munit>-<munit>:[mtype]:value,...,]default_value\n"               \
            "#            value and default_value can be \"auto\""             \
    }

#define UCC_CONFIG_TYPE_PIPELINE_PARAMS                                        \
    {                                                                          \
        ucc_config_sscanf_pipeline_params, ucc_config_sprintf_pipeline_params, \
            ucc_config_clone_pipeline_params,                                  \
            ucc_config_release_pipeline_params, ucs_config_help_generic,       \
            ucs_config_doc_nop,                                                \
            "thresh=<memunit>:fragsize=<memunit>:nfrags="                      \
            "<uint>:pdepth=<uint>:<ordered/parallel/sequential>"               \
    }
#else
#define UCC_CONFIG_TYPE_UINT_RANGED                                            \
    {                                                                          \
        ucc_config_sscanf_uint_ranged, ucc_config_sprintf_uint_ranged,         \
            ucc_config_clone_uint_ranged, ucc_config_release_uint_ranged,      \
            ucs_config_help_generic, "[<munit>-<munit>:[mtype]:value,"         \
            "<munit>-<munit>:[mtype]:value,...,]default_value\n"               \
            "#            value and default_value can be \"auto\""             \
    }

#define UCC_CONFIG_TYPE_PIPELINE_PARAMS                                        \
    {                                                                          \
        ucc_config_sscanf_pipeline_params, ucc_config_sprintf_pipeline_params, \
            ucc_config_clone_pipeline_params,                                  \
            ucc_config_release_pipeline_params, ucs_config_help_generic,       \
            "thresh=<memunit>:fragsize=<memunit>:nfrags="                      \
            "<uint>:pdepth=<uint>:<ordered/parallel/sequential>"               \
    }
#endif

#endif
