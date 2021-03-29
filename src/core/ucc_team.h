/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEAM_H_
#define UCC_TEAM_H_

#include "ucc/api/ucc.h"
#include "utils/ucc_datastruct.h"

typedef struct ucc_context ucc_context_t;
typedef struct ucc_cl_team ucc_cl_team_t;
typedef struct ucc_tl_team ucc_tl_team_t;
typedef struct ucc_coll_task ucc_coll_task_t;
typedef enum {
    UCC_TEAM_SERVICE_TEAM,
    UCC_TEAM_ALLOC_ID,
    UCC_TEAM_CL_CREATE,
} ucc_team_state_t;

typedef struct ucc_team {
    ucc_status_t      status;
    ucc_team_state_t  state;
    ucc_context_t   **contexts;
    uint32_t          num_contexts;
    ucc_team_params_t params;
    ucc_cl_team_t   **cl_teams;
    int               n_cl_teams;
    int               last_team_create_posted;
    uint16_t          id; /*< context-uniq team identifier */
    ucc_rank_t        rank;
    ucc_tl_team_t    *service_team;
    ucc_coll_task_t  *task;
} ucc_team_t;

/* If the bit is set then team_id is provided by the user */
#define UCC_TEAM_ID_EXTERNAL_BIT ((uint16_t)UCC_BIT(15))
#define UCC_TEAM_ID_IS_EXTERNAL(_team) (team->id & UCC_TEAM_ID_EXTERNAL_BIT)
#define UCC_TEAM_ID_MAX ((uint16_t)UCC_BIT(15) - 1)

void ucc_copy_team_params(ucc_team_params_t *dst, const ucc_team_params_t *src);

ucc_status_t ucc_team_destroy_nb(ucc_team_h team);

#endif
