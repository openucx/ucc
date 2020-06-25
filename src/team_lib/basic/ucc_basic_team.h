/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_BASIC_TEAM_H_
#define UCC_BASIC_TEAM_H_
#include "ucc_tl_basic.h"

typedef struct ucc_basic_team_t {
    ucc_tl_team_t  super;
} ucc_basic_team_t;
ucc_status_t ucc_basic_team_create_post(ucc_tl_context_t **context,
                                        uint32_t n_ctxs,
                                        const ucc_team_params_t *params,
                                        ucc_tl_team_t **team);
ucc_status_t ucc_basic_team_create_test(ucc_tl_team_t *team);
ucc_status_t ucc_basic_team_destroy(ucc_tl_team_t *team);
#endif
