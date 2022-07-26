/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_LIB_H_
#define UCC_LIB_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include "components/cl/ucc_cl_type.h"
#include "utils/ucc_parser.h"

typedef struct ucc_cl_lib      ucc_cl_lib_t;
typedef struct ucc_tl_lib      ucc_tl_lib_t;
typedef struct ucc_cl_lib_attr ucc_cl_lib_attr_t;
typedef struct ucc_tl_iface    ucc_tl_iface_t;

typedef struct ucc_lib_config {
    char                    *full_prefix;
    struct {
        ucc_cl_type_t *types;
        unsigned       count;
    } cls;
} ucc_lib_config_t;

typedef struct ucc_lib_info {
    char             *full_prefix;
    int               n_cl_libs_opened;
    int               n_tl_libs_opened;
    ucc_cl_lib_t    **cl_libs;
    ucc_tl_lib_t    **tl_libs;
    ucc_lib_attr_t    attr;
    int               specific_cls_requested;
    ucc_cl_lib_attr_t *cl_attrs;
} ucc_lib_info_t;

void ucc_get_version(unsigned *major_version, unsigned *minor_version,
                     unsigned *release_number);

const char *ucc_get_version_string(void);

int ucc_tl_is_required(ucc_lib_info_t *lib, ucc_tl_iface_t *tl_iface,
                       int forced);
#endif
