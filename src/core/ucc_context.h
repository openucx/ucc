/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCC_CONTEXT_H_
#define UCC_CONTEXT_H_

#include <api/ucc.h>

typedef struct ucc_lib ucc_lib_t;
typedef struct ucc_tl_context ucc_tl_context_t;
typedef struct ucc_tl_context_config ucc_tl_context_config_t;

typedef struct ucc_context {
    ucc_lib_t             *lib;
    ucc_context_params_t  params;
    ucc_tl_context_t      **tl_ctx;
    int                   n_tl_ctx;
} ucc_context_t;

typedef struct ucc_context_config {
    ucc_lib_t               *lib;
    ucc_tl_context_config_t **configs;
    int                     n_tl_cfg;
} ucc_context_config_t;

#endif
