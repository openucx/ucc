/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "cl/ucc_cl_log.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_basic_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_lib_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_lib_config_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_lib_t, &ucc_cl_basic.super, cl_config,
                              UCC_CL_BASIC_DEFAULT_PRIORITY);
    cl_info(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_lib_t)
{
    cl_info(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_cl_basic_lib_t, ucc_cl_lib_t);
