/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEAM_H_
#define UCC_TEAM_H_

#include "ucc/api/ucc.h"

typedef struct ucc_context ucc_context_t;
typedef struct ucc_cl_team ucc_cl_team_t;

typedef struct ucc_team {
    ucc_status_t      status;
    ucc_context_t   **contexts;
    uint32_t          num_contexts;
    ucc_team_params_t params;
    ucc_cl_team_t   **cl_teams;
    int               n_cl_teams;
    int               last_team_create_posted;
    uint16_t          id; /*< context-uniq team identifier */
} ucc_team_t;

void ucc_copy_team_params(ucc_team_params_t *dst, const ucc_team_params_t *src);

ucc_status_t ucc_team_destroy_nb(ucc_team_h team);
#endif
