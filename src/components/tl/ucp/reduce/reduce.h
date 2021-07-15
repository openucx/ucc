/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_H_
#define REDUCE_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"
#include "core/ucc_mc.h"

/* A set of convenience macros used to implement sw based progress
   of the reduce algorithm that uses kn pattern */
enum {
    UCC_REDUCE_KN_PHASE_INIT,
    UCC_REDUCE_KN_PHASE_PROGRESS, /* checks progress */
    UCC_REDUCE_KN_PHASE_MULTI     /* reduce multi after recv from children in current step */
};

#define UCC_REDUCE_KN_CHECK_PHASE(_p)                                          \
    case _p:                                                                   \
        goto _p;                                                               \
        break;

/* UCC_REDUCE_KN_CHECK_PHASE(UCC_REDUCE_KN_PHASE_LOOP);               \ */
#define UCC_REDUCE_KN_GOTO_PHASE(_phase)                                       \
    do {                                                                       \
        switch (_phase) {                                                      \
            UCC_REDUCE_KN_CHECK_PHASE(UCC_REDUCE_KN_PHASE_MULTI);              \
            UCC_REDUCE_KN_CHECK_PHASE(UCC_REDUCE_KN_PHASE_PROGRESS);           \
        case UCC_REDUCE_KN_PHASE_INIT:                                         \
            break;                                                             \
        };                                                                     \
    } while (0)

ucc_status_t ucc_tl_ucp_reduce_init(ucc_tl_ucp_task_t *task);

#endif
