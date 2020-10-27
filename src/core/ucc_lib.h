/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_LIB_H_
#define UCC_LIB_H_

#include "config.h"
#include "api/ucc.h"
#include "utils/ucc_parser.h"

typedef struct ucc_cl_lib ucc_cl_lib_t;
typedef struct ucc_lib_config {
    char                    *full_prefix;
    ucc_config_names_array_t cls;
} ucc_lib_config_t;

typedef struct ucc_lib_info {
    int            n_libs_opened;
    char          *full_prefix;
    ucc_cl_lib_t **libs;
    ucc_lib_attr_t attr;
} ucc_lib_info_t;

void ucc_get_version(unsigned *major_version, unsigned *minor_version,
                     unsigned *release_number);

const char *ucc_get_version_string(void);

#endif
