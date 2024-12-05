/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef GATHER_H_
#define GATHER_H_
#include "tl_ucp_coll.h"
#include "components/mc/ucc_mc.h"

enum {
    UCC_TL_UCP_GATHER_ALG_KNOMIAL,
    UCC_TL_UCP_GATHER_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_gather_algs[UCC_TL_UCP_GATHER_ALG_LAST + 1];

/* A set of convenience macros used to implement sw based progress
   of the gather algorithm that uses kn pattern */
enum
{
    UCC_GATHER_KN_PHASE_INIT,
    UCC_GATHER_KN_PHASE_PROGRESS, /* checks progress */
};

#define UCC_GATHER_KN_CHECK_PHASE(_p)                                          \
    case _p:                                                                   \
        goto _p;

#define UCC_GATHER_KN_GOTO_PHASE(_phase)                                       \
    do {                                                                       \
        switch (_phase) {                                                      \
            UCC_GATHER_KN_CHECK_PHASE(UCC_GATHER_KN_PHASE_PROGRESS);           \
            UCC_GATHER_KN_CHECK_PHASE(UCC_GATHER_KN_PHASE_INIT);               \
        };                                                                     \
    } while (0)

ucc_status_t ucc_tl_ucp_gather_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_gather_knomial_start(ucc_coll_task_t *task);

void ucc_tl_ucp_gather_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_gather_knomial_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_gather_knomial_init_common(ucc_tl_ucp_task_t *task,
                                                   ucc_kn_radix_t radix);

/* Internal interface with custom radix */
ucc_status_t ucc_tl_ucp_gather_knomial_init_r(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *team,
                                              ucc_coll_task_t **task_h,
                                              ucc_kn_radix_t radix);
#endif
