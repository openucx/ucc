/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "cl/ucc_cl_log.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_context_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_context_config_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_context_t, cl_config->cl_lib);
    cl_info(cl_config->cl_lib, "initialized cl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_context_t)
{
    cl_info(self->super.super.lib, "finalizing cl context: %p", self);
}

UCC_CLASS_DEFINE(ucc_cl_basic_context_t, ucc_cl_context_t);
