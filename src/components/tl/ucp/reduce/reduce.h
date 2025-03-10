/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_H_
#define REDUCE_H_
#include "tl_ucp_coll.h"

enum {
    UCC_TL_UCP_REDUCE_ALG_KNOMIAL,
    UCC_TL_UCP_REDUCE_ALG_DBT,
    UCC_TL_UCP_REDUCE_ALG_SRG,
    UCC_TL_UCP_REDUCE_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_reduce_algs[UCC_TL_UCP_REDUCE_ALG_LAST + 1];

#define UCC_TL_UCP_REDUCE_DEFAULT_ALG_SELECT_STR \
    "reduce:0-32K:@0#reduce:32K-inf:@2"

/* A set of convenience macros used to implement sw based progress
   of the reduce algorithm that uses kn pattern */
enum {
    UCC_REDUCE_KN_PHASE_INIT,
    UCC_REDUCE_KN_PHASE_PROGRESS, /* checks progress */
    UCC_REDUCE_KN_PHASE_MULTI     /* reduce multi after recv from children in current step */
};

#define UCC_REDUCE_KN_CHECK_PHASE(_p)                                          \
    case _p:                                                                   \
        goto _p;

#define UCC_REDUCE_KN_GOTO_PHASE(_phase)                                       \
    do {                                                                       \
        switch (_phase) {                                                      \
            UCC_REDUCE_KN_CHECK_PHASE(UCC_REDUCE_KN_PHASE_MULTI);              \
            UCC_REDUCE_KN_CHECK_PHASE(UCC_REDUCE_KN_PHASE_PROGRESS);           \
            UCC_REDUCE_KN_CHECK_PHASE(UCC_REDUCE_KN_PHASE_INIT);               \
        };                                                                     \
    } while (0)


static inline int ucc_tl_ucp_reduce_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_REDUCE_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_reduce_algs[i].name)) {
            break;
        }
    }
    return i;
}

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_ucp_reduce_knomial_start(ucc_coll_task_t *task);

void ucc_tl_ucp_reduce_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_knomial_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_dbt_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *team,
                                        ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_ucp_reduce_srg_knomial_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *team,
                                                ucc_coll_task_t **task_h);

#endif
