/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_
#include "../cl_hier.h"

enum
{
    UCC_CL_HIER_ALLREDUCE_ALG_RAB,
    UCC_CL_HIER_ALLREDUCE_ALG_SPLIT_RAIL,
    UCC_CL_HIER_ALLREDUCE_ALG_LAST,
};

extern ucc_base_coll_alg_info_t
    ucc_cl_hier_allreduce_algs[UCC_CL_HIER_ALLREDUCE_ALG_LAST + 1];

#define UCC_CL_HIER_ALLREDUCE_DEFAULT_ALG_SELECT_STR "allreduce:0-4k:@rab"

ucc_status_t ucc_cl_hier_allreduce_rab_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t      *team,
                                            ucc_coll_task_t     **task);

ucc_status_t
ucc_cl_hier_allreduce_split_rail_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_coll_task_t     **task);

static inline int ucc_cl_hier_allreduce_alg_from_str(const char *str)
{
    int i;

    for (i = 0; i < UCC_CL_HIER_ALLREDUCE_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_cl_hier_allreduce_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
