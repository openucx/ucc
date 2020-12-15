/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_CL_H_
#define UCC_CL_H_

#include "components/base/ucc_base_iface.h"
#include "ucc_cl_type.h"

typedef struct ucc_cl_lib     ucc_cl_lib_t;
typedef struct ucc_cl_iface   ucc_cl_iface_t;
typedef struct ucc_cl_context ucc_cl_context_t;

typedef struct ucc_cl_lib_config {
    ucc_base_config_t  super;
    ucc_cl_iface_t    *iface;
    int                priority;
} ucc_cl_lib_config_t;
extern ucc_config_field_t ucc_cl_lib_config_table[];


typedef struct ucc_cl_context_config {
    ucc_base_config_t super;
    ucc_cl_lib_t   *cl_lib;
} ucc_cl_context_config_t;
extern ucc_config_field_t ucc_cl_context_config_table[];

ucc_status_t ucc_cl_context_config_read(ucc_cl_lib_t *cl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_cl_context_config_t **cl_config);

ucc_status_t ucc_cl_lib_config_read(ucc_cl_iface_t *iface, const char *full_prefix,
                                    const ucc_lib_config_t *config,
                                    ucc_cl_lib_config_t **cl_config);

typedef struct ucc_cl_iface {
    ucc_component_iface_t          super;
    ucc_cl_type_t                  type;
    ucc_lib_attr_t                 attr;
    ucc_config_global_list_entry_t cl_lib_config;
    ucc_config_global_list_entry_t cl_context_config;
    ucc_base_lib_iface_t           lib;
    ucc_base_context_iface_t       context;
} ucc_cl_iface_t;

typedef struct ucc_cl_lib {
    ucc_base_lib_t              super;
    ucc_cl_iface_t             *iface;
    int                         priority;
} ucc_cl_lib_t;

UCC_CLASS_DECLARE(ucc_cl_lib_t, ucc_cl_iface_t *, const ucc_cl_lib_config_t *,
                  int);

typedef struct ucc_cl_context {
    ucc_base_context_t super;
} ucc_cl_context_t;

UCC_CLASS_DECLARE(ucc_cl_context_t, ucc_cl_lib_t *);

#define UCC_CL_IFACE_NAME_PREFIX(_NAME, _cfg)                                  \
    .name   = UCC_PP_MAKE_STRING(CL_##_NAME) " " UCC_PP_MAKE_STRING(_cfg),     \
    .prefix = UCC_PP_MAKE_STRING(CL_##_NAME##_)

#define UCC_CL_IFACE_CFG(_cfg, _name, _NAME)                                   \
    .super.cl_##_cfg##_config = {                                              \
        UCC_CL_IFACE_NAME_PREFIX(_NAME, _cfg),                                 \
        .table = ucc_cl_##_name##_##_cfg##_config_table,                       \
        .size  = sizeof(ucc_cl_##_name##_##_cfg##_config_t)}

#define UCC_CL_IFACE_DECLARE(_name, _NAME)                                     \
    ucc_cl_##_name##_iface_t ucc_cl_##_name = {                                \
        UCC_CL_IFACE_CFG(lib, _name, _NAME),                                   \
        UCC_CL_IFACE_CFG(context, _name, _NAME),                               \
        .super.super.name = UCC_PP_MAKE_STRING(_name),                         \
        .super.type       = UCC_CL_##_NAME,                                    \
        .super.lib.init   = UCC_CLASS_NEW_FUNC_NAME(ucc_cl_##_name##_lib_t),   \
        .super.lib.finalize =                                                  \
            UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_##_name##_lib_t),                \
        .super.context.create =                                                \
            UCC_CLASS_NEW_FUNC_NAME(ucc_cl_##_name##_context_t),               \
        .super.context.destroy =                                               \
            UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_##_name##_context_t)};           \
    UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_cl_##_name.super.cl_lib_config,       \
                                    &ucc_config_global_list);                  \
    UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_cl_##_name.super.cl_context_config,   \
                                    &ucc_config_global_list)

#endif
