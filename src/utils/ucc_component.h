/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_COMPONENT_H_
#define UCC_COMPONENT_H_

#include "config.h"
#include <api/ucc.h>

typedef struct ucc_component_iface {
    void *dl_handle;
} ucc_component_iface_t;

ucc_status_t ucc_components_load(const char* framework_name,
                                 ucc_component_iface_t ***components, int *n_components);
#endif
