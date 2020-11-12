/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_CL_TYPE_H_
#define UCC_CL_TYPE_H_

typedef enum {
    UCC_CL_BASIC,
    UCC_CL_ALL,
    UCC_CL_LAST
} ucc_cl_type_t;

extern const char *ucc_cl_names[];

#endif
