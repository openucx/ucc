/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_H_
#define ALLTOALL_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE,
    UCC_TL_UCP_ALLTOALL_ALG_BRUCK,
    UCC_TL_UCP_ALLTOALL_ALG_ONESIDED,
    UCC_TL_UCP_ALLTOALL_ALG_ONESIDED_CA,
    UCC_TL_UCP_ALLTOALL_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_ucp_alltoall_algs[UCC_TL_UCP_ALLTOALL_ALG_LAST + 1];

#define UCC_TL_UCP_ALLTOALL_DEFAULT_ALG_SELECT_STR_PATTERN \
"alltoall:host:0-%d:@bruck"

char* ucc_tl_ucp_alltoall_score_str_get(ucc_tl_ucp_team_t *team);

ucc_status_t ucc_tl_ucp_alltoall_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_common(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_ucp_alltoall_bruck_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task_h);


ucc_status_t ucc_tl_ucp_alltoall_onesided_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_ucp_alltoall_onesided_ca_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *team,
                                                  ucc_coll_task_t **task_h);


#define ALLTOALL_CHECK_INPLACE(_args, _team)                                   \
    do {                                                                       \
        if (UCC_IS_INPLACE(_args)) {                                           \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "inplace alltoall is not supported");                     \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

#define ALLTOALL_CHECK_USERDEFINED_DT(_args, _team  )                          \
    do {                                                                       \
        if (!ucc_coll_args_is_predefined_dt(&(_args), UCC_RANK_INVALID)) {     \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "user defined datatype is not supported");                \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

#define ALLTOALL_TASK_CHECK(_args, _team)                                      \
    ALLTOALL_CHECK_INPLACE((_args), (_team));                                  \
    ALLTOALL_CHECK_USERDEFINED_DT((_args), (_team));

static inline int ucc_tl_ucp_alltoall_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLTOALL_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_alltoall_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
