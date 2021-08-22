/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef SCATTER_H_
#define SCATTER_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

/* A set of convenience macros used to implement sw based progress
   of the scatter algorithm that uses kn pattern */
//enum {
//    UCC_SCATTER_KN_PHASE_INIT,
//    UCC_SCATTER_KN_PHASE_LOOP, /* main loop of recursive k-ing */
//};

/*
#define UCC_SCATTER_KN_CHECK_PHASE(_p)                                        \
    case _p:                                                                  \
        goto _p;

#define UCC_SCATTER_KN_GOTO_PHASE(_phase)                                     \
    do {                                                                      \
        switch (_phase) {                                                     \
            UCC_SCATTER_KN_CHECK_PHASE(UCC_SCATTER_KN_PHASE_LOOP);            \
        case UCC_SCATTER_KN_PHASE_INIT:                                       \
            break;                                                            \
        };                                                                    \
    } while (0)
    */

/* Base interface signature: uses scatter_kn_radix from config. */

ucc_status_t
ucc_tl_ucp_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                ucc_base_team_t      *team,
                                ucc_coll_task_t     **task_h);

/* Internal interface to KN scatter with custom radix */
ucc_status_t ucc_tl_ucp_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix);
#endif
