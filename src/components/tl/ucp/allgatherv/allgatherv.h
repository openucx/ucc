/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHERV_H_
#define ALLGATHERV_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLGATHERV_ALG_RING,
    UCC_TL_UCP_ALLGATHERV_ALG_KNOMIAL,
    UCC_TL_UCP_ALLGATHERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_allgatherv_algs[UCC_TL_UCP_ALLGATHERV_ALG_LAST + 1];

#define UCC_TL_UCP_ALLGATHERV_DEFAULT_ALG_SELECT_STR                           \
    "allgatherv:0-4k:@knomial#allgatherv:4k-inf:@ring"

char *ucc_tl_ucp_allgatherv_score_str_get(ucc_tl_ucp_team_t *team);

static inline int ucc_tl_ucp_allgatherv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLGATHERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_allgatherv_algs[i].name)) {
            break;
        }
    }
    return i;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_init_common(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_ring_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team,
                                             ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_ucp_allgatherv_knomial_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *team,
                                                ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task);
#endif
