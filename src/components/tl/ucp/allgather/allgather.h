/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef ALLGATHER_H_
#define ALLGATHER_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL,
    UCC_TL_UCP_ALLGATHER_ALG_RING,
    UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR,
    UCC_TL_UCP_ALLGATHER_ALG_BRUCK,
    UCC_TL_UCP_ALLGATHER_ALG_SPARBIT,
    UCC_TL_UCP_ALLGATHER_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgather_algs[UCC_TL_UCP_ALLGATHER_ALG_LAST + 1];

#define UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR                            \
    "allgather:0-4k:@0#allgather:4k-inf:@%d"

#define UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR_1PPN                       \
    "allgather:0-4k:@0#allgather:4k-inf:%s:@0#allgather:4k-inf:%s:@%d"

#define UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR_1PPN_CUDA                  \
    "allgather:0-4k:@0#allgather:4k-inf:%s:@0#allgather:4k-inf:@%d"

char *ucc_tl_ucp_allgather_score_str_get(ucc_tl_ucp_team_t *team);

static inline int ucc_tl_ucp_allgather_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLGATHER_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_allgather_algs[i].name)) {
            break;
        }
    }
    return i;
}

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task);

/* Ring */
ucc_status_t ucc_tl_ucp_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t      *team,
                                            ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_ucp_allgather_ring_init_common(ucc_tl_ucp_task_t *task);

void ucc_tl_ucp_allgather_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_ring_start(ucc_coll_task_t *task);

/* Neighbor Exchange */
ucc_status_t ucc_tl_ucp_allgather_neighbor_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

void ucc_tl_ucp_allgather_neighbor_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_neighbor_start(ucc_coll_task_t *task);

/* Bruck */
ucc_status_t ucc_tl_ucp_allgather_bruck_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

void ucc_tl_ucp_allgather_bruck_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_bruck_start(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_bruck_finalize(ucc_coll_task_t *coll_task);

/* Sparbit */
ucc_status_t ucc_tl_ucp_allgather_sparbit_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

/* Uses allgather_kn_radix from config */
ucc_status_t ucc_tl_ucp_allgather_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h);

/* Internal interface with custom radix */
ucc_status_t ucc_tl_ucp_allgather_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix);
#endif
