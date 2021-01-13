/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_team_t, ucc_base_context_t *cl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_cl_basic_context_t *ctx =
        ucc_derived_of(cl_context, ucc_cl_basic_context_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super);
    cl_info(cl_context->lib, "initialized cl team: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_team_t)
{
    cl_info(self->super.super.context->lib, "finalizing cl team: %p", self);
}

ucc_status_t ucc_cl_basic_team_create_test(ucc_base_team_t *cl_team)
{
    /* ucc_cl_basic_team_t *team = ucc_derived_of(cl_team, ucc_cl_basic_team_t); */
    return UCC_OK;
}

UCC_CLASS_DEFINE(ucc_cl_basic_team_t, ucc_cl_team_t);
