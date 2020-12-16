/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_LIB_H_
#define UCC_LIB_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include "components/cl/ucc_cl_type.h"
#include "utils/ucc_parser.h"

typedef struct ucc_cl_lib ucc_cl_lib_t;
typedef struct ucc_lib_config {
    char                    *full_prefix;
    struct {
        ucc_cl_type_t *types;
        unsigned       count;
    } cls;
} ucc_lib_config_t;

typedef struct ucc_lib_info {
    int            n_libs_opened;
    char          *full_prefix;
    ucc_cl_lib_t **libs;
    ucc_lib_attr_t attr;
    int            specific_cls_requested;
} ucc_lib_info_t;

void ucc_get_version(unsigned *major_version, unsigned *minor_version,
                     unsigned *release_number);

const char *ucc_get_version_string(void);

#define UCC_COPY_PARAM_BY_FIELD(_dst, _src, _FIELD, _field)                    \
    do {                                                                       \
        if ((_src)->mask & (_FIELD)) {                                         \
            (_dst)->_field = (_src)->_field;                                   \
        }                                                                      \
    } while (0)

#endif
