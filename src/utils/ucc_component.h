/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_COMPONENT_H_
#define UCC_COMPONENT_H_

#include "config.h"
#include <api/ucc.h>

#define UCC_MAX_FRAMEWORK_NAME_LEN 64
#define UCC_MAX_COMPONENT_NAME_LEN 64

typedef struct ucc_component_iface {
    void *dl_handle;
} ucc_component_iface_t;

typedef struct ucc_component_framework {
    char                   *framework_name;
    int                     n_components;
    ucc_component_iface_t **components;
} ucc_component_framework_t;

/* ucc_components_load searches for all available dynamic components 
   with the name matching the pattern: ucc_<framework_name>_*.so. 
   The search is performed in the ucc_global_config.component_path.
   Each dynamic component must have a component interface structure defined.
   This structure must inherit from ucc_component_iface_t.
   The name of the structure must follow the pattern:
   ucc_<framework_name>_<component_name>. */
ucc_status_t ucc_components_load(const char *framework_name,
                                 ucc_component_framework_t *framework);
#endif
