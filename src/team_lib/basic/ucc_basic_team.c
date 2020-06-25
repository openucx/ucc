/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "ucc_basic_context.h"
#include "ucc_basic_team.h"
#include <stdio.h>


ucc_status_t ucc_basic_team_create_post(ucc_tl_context_t **contexts,
                                        uint32_t n_ctxs,
                                        const ucc_team_params_t *params,
                                        ucc_tl_team_t **team)
{
    ucc_status_t     status = UCC_OK;
    ucc_basic_team_t *basic_team;
    basic_team = (ucc_basic_team_t*)malloc(sizeof(ucc_basic_team_t));
    ucc_tl_info(contexts[0]->tl_lib, "create_post tl team %p\n", basic_team);
    *team = &basic_team->super;
    return UCC_OK;
}

ucc_status_t ucc_basic_team_create_test(ucc_tl_team_t *team)
{
    ucc_basic_team_t *basic_team = ucs_derived_of(team, ucc_basic_team_t);
    ucc_status_t     status      = UCC_OK;
    if (status == UCC_OK) {
        ucc_tl_info(team->tl_lib, "tl team %p create complete\n", team);
    }
    return status;
}

ucc_status_t ucc_basic_team_destroy(ucc_tl_team_t *team)
{
    ucc_basic_team_t *basic_team = ucs_derived_of(team, ucc_basic_team_t);
    ucc_status_t     status      = UCC_OK;
    free(basic_team);
    return UCC_OK;
}
