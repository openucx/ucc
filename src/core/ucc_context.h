/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_CONTEXT_H_
#define UCC_CONTEXT_H_

#include "ucc/api/ucc.h"

typedef struct ucc_lib_info          ucc_lib_info_t;
typedef struct ucc_cl_context        ucc_cl_context_t;
typedef struct ucc_cl_context_config ucc_cl_context_config_t;

typedef struct ucc_context {
    ucc_lib_info_t      *lib;
    ucc_context_params_t params;
    ucc_context_attr_t   attr;
    ucc_cl_context_t   **cl_ctx;
    int                  n_cl_ctx;
} ucc_context_t;

typedef struct ucc_context_config {
    ucc_lib_info_t           *lib;
    ucc_cl_context_config_t **configs;
    int                       n_cl_cfg;
} ucc_context_config_t;

#endif
