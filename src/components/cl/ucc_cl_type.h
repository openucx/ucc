/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_CL_TYPE_H_
#define UCC_CL_TYPE_H_

#include <string.h>

typedef enum {
    UCC_CL_BASIC,
    UCC_CL_HIER,
    UCC_CL_ALL,
    UCC_CL_LAST
} ucc_cl_type_t;

extern const char *ucc_cl_names[];

static inline ucc_cl_type_t ucc_cl_name_to_type(const char *cl_name)
{
    int i;
    for (i = 0; i < UCC_CL_LAST; i++) {
        if (0 == strcmp(cl_name, ucc_cl_names[i])) {
            break;
        }
    }
    return (ucc_cl_type_t)i;
}

/* takes string of comma separated cls and returns and array of ucc_cl_type_t.
   the checks for correct input are done.
   the allocated should be freed by the user. */
ucc_status_t ucc_parse_cls_string(const char *cls_str,
                                  ucc_cl_type_t **cls_array, int *n_cls);
#endif
