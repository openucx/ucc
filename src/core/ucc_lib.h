/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_LIB_H_
#define UCC_LIB_H_

#include "config.h"
#include <api/ucc.h>

typedef struct ucc_lib_config {
    const char *tls;
} ucc_lib_config_t;

typedef struct ucc_team_lib ucc_team_lib_t;
typedef struct ucc_lib {
    int            n_libs_opened;
    int            libs_array_size;
    ucc_team_lib_t **libs;
} ucc_lib_t;

#endif
