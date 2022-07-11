/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef BCAST_H_
#define BCAST_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_BCAST_ALG_KNOMIAL,
    UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL,
    UCC_TL_UCP_BCAST_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_bcast_algs[UCC_TL_UCP_BCAST_ALG_LAST + 1];

#define UCC_TL_UCP_BCAST_DEFAULT_ALG_SELECT_STR \
    "bcast:0-32k:@0#bcast:32k-inf:@1"

static inline int ucc_tl_ucp_bcast_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_BCAST_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_bcast_algs[i].name)) {
            break;
        }
    }
    return i;
}

ucc_status_t ucc_tl_ucp_bcast_init(ucc_tl_ucp_task_t *task);

ucc_status_t
ucc_tl_ucp_bcast_knomial_init(ucc_base_coll_args_t *coll_args,
                              ucc_base_team_t *team, ucc_coll_task_t **task_h);
void
ucc_tl_ucp_bcast_knomial_progress(ucc_coll_task_t *task);

ucc_status_t
ucc_tl_ucp_bcast_knomial_start(ucc_coll_task_t *task);

ucc_status_t
ucc_tl_ucp_bcast_sag_knomial_init(ucc_base_coll_args_t *coll_args,
                              ucc_base_team_t *team, ucc_coll_task_t **task_h);

#endif
