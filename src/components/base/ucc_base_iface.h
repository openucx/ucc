/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_BASE_IFACE_H_
#define UCC_BASE_IFACE_H_
#include "ucc/api/ucc.h"
#include "core/ucc_lib.h"
#include "core/ucc_context.h"
#include "utils/ucc_component.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_class.h"
#include "utils/ucc_malloc.h"

typedef struct ucc_base_lib {
    ucc_log_component_config_t log_component;
} ucc_base_lib_t;

typedef struct ucc_base_config {
    ucc_config_global_list_entry_t *cfg_entry;
    ucc_log_component_config_t      log_component;
} ucc_base_config_t;

typedef struct ucc_base_lib_params {
    ucc_lib_params_t params;
} ucc_base_lib_params_t;
extern ucc_config_field_t ucc_base_config_table[];

typedef struct ucc_base_lib_iface {
    ucc_status_t (*init)(const ucc_base_lib_params_t *params,
                         const ucc_base_config_t *config, ucc_base_lib_t **lib);
    void         (*finalize)(ucc_base_lib_t *lib);
} ucc_base_lib_iface_t;

typedef struct ucc_base_context_params {
    ucc_context_params_t params;
    int                  estimated_num_eps;
    int                  estimated_num_ppn;
    ucc_thread_mode_t    thread_mode;
    const char          *prefix;
} ucc_base_context_params_t;

typedef struct ucc_base_context {
    ucc_base_lib_t *lib;
} ucc_base_context_t;

typedef struct ucc_base_context_iface {
    ucc_status_t (*create)(const ucc_base_context_params_t *params,
                           const ucc_base_config_t *config,
                           ucc_base_context_t **ctx);
    void         (*destroy)(ucc_base_context_t *ctx);
} ucc_base_context_iface_t;

ucc_status_t ucc_base_config_read(const char *full_prefix,
                                  ucc_config_global_list_entry_t *cfg_entry,
                                  ucc_base_config_t **config);

static inline void ucc_base_config_release(ucc_base_config_t *config)
{
    ucc_config_parser_release_opts(config, config->cfg_entry->table);
    ucc_free(config);
}

#define UCC_IFACE_NAME_PREFIX(_F, _NAME, _cfg)                                 \
    .name   = UCC_PP_MAKE_STRING(_F##_NAME) " " UCC_PP_MAKE_STRING(_cfg),      \
    .prefix = UCC_PP_MAKE_STRING(_F##_NAME##_)

#define UCC_IFACE_CFG(_F, _f, _cfg, _name, _NAME)                              \
    .super._f##_cfg##_config = {                                               \
        UCC_IFACE_NAME_PREFIX(_F, _NAME, _cfg),                                \
        .table = ucc_##_f##_name##_##_cfg##_config_table,                      \
        .size  = sizeof(ucc_##_f##_name##_##_cfg##_config_t)}

#define UCC_BASE_IFACE_DECLARE(_F, _f, _name, _NAME, ...)                      \
    ucc_##_f##_name##_iface_t ucc_##_f##_name = {                              \
        UCC_IFACE_CFG(_F, _f, lib, _name, _NAME),                              \
        UCC_IFACE_CFG(_F, _f, context, _name, _NAME),                          \
        .super.super.name = UCC_PP_MAKE_STRING(_name),                         \
        __VA_ARGS__                                                            \
        .super.lib.init   = UCC_CLASS_NEW_FUNC_NAME(ucc_##_f##_name##_lib_t),  \
        .super.lib.finalize =                                                  \
            UCC_CLASS_DELETE_FUNC_NAME(ucc_##_f##_name##_lib_t),               \
        .super.context.create =                                                \
            UCC_CLASS_NEW_FUNC_NAME(ucc_##_f##_name##_context_t),              \
        .super.context.destroy =                                               \
        UCC_CLASS_DELETE_FUNC_NAME(ucc_##_f##_name##_context_t)};              \
    UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_##_f##_name.super._f##lib_config,     \
                                    &ucc_config_global_list);                  \
    UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_##_f##_name.super._f##context_config, \
                                    &ucc_config_global_list)

#endif
