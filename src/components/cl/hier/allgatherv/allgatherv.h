/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHERV_H_
#define ALLGATHERV_H_
#include "../cl_hier.h"

enum
{
    UCC_CL_HIER_ALLGATHERV_ALG_GAB,
    UCC_CL_HIER_ALLGATHERV_ALG_LAST,
};

extern ucc_base_coll_alg_info_t
    ucc_cl_hier_allgatherv_algs[UCC_CL_HIER_ALLGATHERV_ALG_LAST + 1];

#define UCC_CL_HIER_ALLGATHERV_DEFAULT_ALG_SELECT_STR "allgatherv:0-2k:host:@gab"

ucc_status_t ucc_cl_hier_allgatherv_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t       *team,
                                        ucc_coll_task_t      **task);

static inline int ucc_cl_hier_allgatherv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_CL_HIER_ALLGATHERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_cl_hier_allgatherv_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
