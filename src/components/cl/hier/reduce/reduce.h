/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef REDUCE_H_
#define REDUCE_H_
#include "../cl_hier.h"

enum
{
    UCC_CL_HIER_REDUCE_ALG_2STEP,
    UCC_CL_HIER_REDUCE_ALG_LAST,
};

extern ucc_base_coll_alg_info_t
    ucc_cl_hier_reduce_algs[UCC_CL_HIER_REDUCE_ALG_LAST + 1];

#define UCC_CL_HIER_REDUCE_DEFAULT_ALG_SELECT_STR "reduce:0-4k:@2step"

ucc_status_t ucc_cl_hier_reduce_2step_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t *team,
                                           ucc_coll_task_t **task);

static inline int ucc_cl_hier_reduce_alg_from_str(const char *str)
{
    int i;

    for (i = 0; i < UCC_CL_HIER_REDUCE_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_cl_hier_reduce_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
