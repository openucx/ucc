/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_BASIC_H_
#define UCC_CL_BASIC_H_
#include "cl/ucc_cl.h"

typedef struct ucc_cl_basic_iface {
    ucc_cl_iface_t super;
} ucc_cl_basic_iface_t;
/* Extern iface should follow the pattern: ucc_cl_<cl_name> */
extern ucc_cl_basic_iface_t ucc_cl_basic;

typedef struct ucc_cl_basic_lib_config {
    ucc_cl_lib_config_t super;
} ucc_cl_basic_lib_config_t;

typedef struct ucc_cl_basic_lib {
    ucc_cl_lib_t super;
} ucc_cl_basic_lib_t;

UCC_CLASS_DECLARE(ucc_cl_basic_lib_t, ucc_cl_iface_t *,
                  const ucc_lib_config_t *, const ucc_cl_lib_config_t *);
#endif
