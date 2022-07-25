/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_SCATTER_H_
#define REDUCE_SCATTER_H_
#include "../tl_ucp_reduce.h"

enum
{
    UCC_TL_UCP_REDUCE_SCATTER_ALG_RING,
    UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_scatter_algs[UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST + 1];

#define UCC_TL_UCP_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR                       \
    "reduce_scatter:@ring"

static inline int ucc_tl_ucp_reduce_scatter_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_reduce_scatter_algs[i].name)) {
            break;
        }
    }
    return i;
}

/* Base interface signature: uses reduce_scatter_kn_radix from config. */

ucc_status_t
ucc_tl_ucp_reduce_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t      *team,
                                       ucc_coll_task_t     **task_h);

/* Internal interface to KN reduce scatter with custom radix */
ucc_status_t ucc_tl_ucp_reduce_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix);

ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *     team,
                                    ucc_coll_task_t **    task_h);
#endif
