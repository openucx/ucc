/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_BASIC_H_
#define UCC_CL_BASIC_H_
#include "components/cl/ucc_cl.h"
#include "components/cl/ucc_cl_log.h"
#include "components/tl/ucc_tl.h"

#define UCC_CL_BASIC_DEFAULT_PRIORITY 10

typedef struct ucc_cl_basic_iface {
    ucc_cl_iface_t super;
} ucc_cl_basic_iface_t;
/* Extern iface should follow the pattern: ucc_cl_<cl_name> */
extern ucc_cl_basic_iface_t ucc_cl_basic;

typedef struct ucc_cl_basic_lib_config {
    ucc_cl_lib_config_t super;
} ucc_cl_basic_lib_config_t;

typedef struct ucc_cl_basic_context_config {
    ucc_cl_context_config_t super;
    int                     test_param;
} ucc_cl_basic_context_config_t;

typedef struct ucc_cl_basic_lib {
    ucc_cl_lib_t super;
} ucc_cl_basic_lib_t;
UCC_CLASS_DECLARE(ucc_cl_basic_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_basic_context {
    ucc_cl_context_t super;
    ucc_tl_context_t *tl_ucp_ctx;
} ucc_cl_basic_context_t;
UCC_CLASS_DECLARE(ucc_cl_basic_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

#endif
