/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEAM_H_
#define UCC_TEAM_H_

#include <api/ucc.h>
#include <ucc_context.h>
#include "team_lib/ucc_tl.h"

/* TODO compute from ucc_coll_type_t */
#define UCC_COLL_LAST 10

#define UCC_CHECK_TEAM(_team)                                                    \
    do {                                                                         \
        if (_team->status != UCC_OK) {                                           \
            ucc_error("team %p is used before team_create is completed", _team); \
            return UCC_ERR_INVALID_PARAM;                                        \
        }                                                                        \
    } while(0)

typedef struct ucc_team {
    ucc_context_t  **contexts;
    ucc_tl_iface_t *iface;
    int            n_ctx;
    int            coll_team_id[UCC_COLL_LAST];
    int            n_tl_teams;
    ucc_status_t   status;
    ucc_tl_team_t  *tl_teams[1];
} ucc_team_t;

#endif
