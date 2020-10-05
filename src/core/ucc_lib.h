/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_LIB_H_
#define UCC_LIB_H_

#include "config.h"
#include <api/ucc.h>
#include <ucs/config/types.h>
typedef struct ucc_lib_config {
    char                     *full_prefix;
    ucs_config_names_array_t ccms;
} ucc_lib_config_t;

typedef struct ucc_ccm_lib ucc_ccm_lib_t;
typedef struct ucc_ccm_iface ucc_ccm_iface_t;

typedef struct ucc_lib_info {
    int           n_libs_opened;
    char          *full_prefix;
    ucc_ccm_lib_t **libs;
} ucc_lib_info_t;

struct ucc_static_lib_data {
    int             n_ccms_loaded;
    ucc_ccm_iface_t **ccm_ifaces;
};
extern struct ucc_static_lib_data ucc_lib_data;

#endif

