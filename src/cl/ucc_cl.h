/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_CL_H_
#define UCC_CL_H_

#include "ucc/api/ucc.h"
#include "core/ucc_lib.h"
#include "ucc_cl_type.h"
#include "utils/ucc_component.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_class.h"

typedef struct ucc_cl_lib     ucc_cl_lib_t;
typedef struct ucc_cl_iface   ucc_cl_iface_t;
typedef struct ucc_cl_context ucc_cl_context_t;

typedef struct ucc_cl_lib_config {
    /* Log level above which log messages will be printed */
    ucc_log_component_config_t log_component;
    /* Team library priority */
    int                        priority;
} ucc_cl_lib_config_t;

typedef struct ucc_cl_context_config {
    ucc_cl_iface_t *iface;
    ucc_cl_lib_t   *cl_lib;
} ucc_cl_context_config_t;

extern ucc_config_field_t ucc_cl_lib_config_table[];
extern ucc_config_field_t ucc_cl_context_config_table[];

typedef struct ucc_cl_iface {
    ucc_component_iface_t          super;
    ucc_cl_type_t                  type;
    int                            priority;
    ucc_lib_attr_t                 attr;
    ucc_config_global_list_entry_t cl_lib_config;
    ucs_config_global_list_entry_t cl_context_config;
    ucc_status_t                   (*init)(const ucc_lib_params_t *params,
                                           const ucc_lib_config_t *config,
                                           const ucc_cl_lib_config_t *cl_config,
                                           ucc_cl_lib_t **cl_lib);
    ucc_status_t                   (*finalize)(ucc_cl_lib_t *cl_lib);
} ucc_cl_iface_t;
UCC_CLASS_DECLARE(ucc_cl_iface_t);

typedef struct ucc_cl_lib {
    ucc_cl_iface_t             *iface;
    ucc_log_component_config_t  log_component;
    int                         priority;
} ucc_cl_lib_t;

UCC_CLASS_DECLARE(ucc_cl_lib_t, ucc_cl_iface_t *, const ucc_lib_config_t *,
                  const ucc_cl_lib_config_t *);

typedef struct ucc_cl_context {
    ucc_cl_lib_t *cl_lib;
} ucc_cl_context_t;

#define UCC_CL_IFACE_NAME_PREFIX(_NAME)                 \
    .name   = UCC_PP_MAKE_STRING(CL_ ## _NAME),         \
    .prefix = UCC_PP_MAKE_STRING(CL_ ## _NAME ##_)

#define UCC_CL_IFACE_CFG(_cfg, _name, _NAME)                            \
    .super.cl_ ## _cfg ## _config = {                                   \
        UCC_CL_IFACE_NAME_PREFIX(_NAME),                                \
        .table  = ucc_cl_ ## _name ## _ ## _cfg ## _config_table,       \
        .size   = sizeof(ucc_cl_ ## _name ## _ ## _cfg ## _config_t)}

#define UCC_CL_IFACE_DECLARE(_name, _NAME, _priority)           \
    ucc_cl_ ## _name ## _iface_t ucc_cl_ ## _name = {           \
        UCC_CL_IFACE_CFG(lib, _name, _NAME),                    \
        UCC_CL_IFACE_CFG(context, _name, _NAME),                \
        .super.super.name = UCC_PP_MAKE_STRING(basic),          \
        .super.type       = UCC_CL_ ## _NAME,                   \
        .super.priority   = _priority,                          \
        .super.init       = ucc_cl_ ## _name ## _lib_init,      \
        .super.finalize   = ucc_cl_ ## _name ## _lib_finalize,  \
    }

#endif
