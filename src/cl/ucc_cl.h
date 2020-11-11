/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_CL_H_
#define UCC_CL_H_

#include "api/ucc.h"
#include "core/ucc_lib.h"
#include "ucc_cl_type.h"
#include "utils/ucc_component.h"
#include "utils/ucc_parser.h"

typedef struct ucc_cl_lib   ucc_cl_lib_t;
typedef struct ucc_cl_iface ucc_cl_iface_t;

typedef struct ucc_cl_lib_config {
    /* Log level above which log messages will be printed */
    ucc_log_component_config_t log_component;
    /* Team library priority */
    int                        priority;
} ucc_cl_lib_config_t;

extern ucc_config_field_t ucc_cl_lib_config_table[];

typedef struct ucc_cl_iface {
    ucc_component_iface_t          super;
    ucc_cl_type_t                  type;
    int                            priority;
    ucc_lib_attr_t                 attr;
    ucc_config_global_list_entry_t cl_lib_config;
    ucc_status_t                   (*init)(const ucc_lib_params_t *params,
                                           const ucc_lib_config_t *config,
                                           const ucc_cl_lib_config_t *cl_config,
                                           ucc_cl_lib_t **cl_lib);
    ucc_status_t                   (*finalize)(ucc_cl_lib_t *cl_lib);
} ucc_cl_iface_t;

typedef struct ucc_cl_lib {
    ucc_cl_iface_t             *iface;
    ucc_log_component_config_t  log_component;
    int                         priority;
} ucc_cl_lib_t;

/* Every component should call this init function during (*init) in order
   to use common priority/logging mechanism */
static inline void ucc_cl_lib_init(ucc_cl_lib_t *cl_lib, ucc_cl_iface_t *cl_iface,
                                   const ucc_cl_lib_config_t *cl_config)
{
    cl_lib->log_component = cl_config->log_component;
    ucc_strncpy_safe(cl_lib->log_component.name, cl_iface->cl_lib_config.name,
                     sizeof(cl_lib->log_component.name));
    cl_lib->priority =
        (-1 == cl_config->priority) ? cl_iface->priority : cl_config->priority;
}

#endif
