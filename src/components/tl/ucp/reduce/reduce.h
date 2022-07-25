/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_H_
#define REDUCE_H_
#include "../tl_ucp_reduce.h"

enum {
    UCC_TL_UCP_REDUCE_ALG_KNOMIAL,
    UCC_TL_UCP_REDUCE_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_reduce_algs[UCC_TL_UCP_REDUCE_ALG_LAST + 1];

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

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_knomial_start(ucc_coll_task_t *task);

void ucc_tl_ucp_reduce_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_knomial_finalize(ucc_coll_task_t *task);

#endif
