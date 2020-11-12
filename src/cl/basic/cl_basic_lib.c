/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "cl/ucc_cl_log.h"
#include "utils/ucc_malloc.h"

static ucc_config_field_t ucc_cl_basic_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_basic_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {NULL}
};

static ucs_config_field_t ucc_cl_basic_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_basic_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {"TEST_PARAM", "5", "For dbg test purpuse : don't commit",
     ucc_offsetof(ucc_cl_basic_context_config_t, test_param),
     UCC_CONFIG_TYPE_UINT},

    {NULL}
};

UCC_CLASS_INIT_FUNC(ucc_cl_basic_lib_t, ucc_cl_iface_t *cl_iface,
                    const ucc_lib_config_t *config,
                    const ucc_cl_lib_config_t *cl_config)
{
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_lib_t, cl_iface, config, cl_config);
    cl_info(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_lib_t)
{
    cl_info(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_cl_basic_lib_t, ucc_cl_lib_t);

static ucc_status_t ucc_cl_basic_lib_init(const ucc_lib_params_t *params,
                                          const ucc_lib_config_t *config,
                                          const ucc_cl_lib_config_t *cl_config,
                                          ucc_cl_lib_t **cl_lib)
{
    return UCC_CLASS_NEW(ucc_cl_basic_lib_t, cl_lib, &ucc_cl_basic.super,
                         config, cl_config);
}

static ucc_status_t ucc_cl_basic_lib_finalize(ucc_cl_lib_t *cl_lib)
{
    ucc_cl_basic_lib_t *lib = ucc_derived_of(cl_lib, ucc_cl_basic_lib_t);
    UCC_CLASS_DELETE(ucc_cl_basic_lib_t, lib);
    return UCC_OK;
}

ucc_cl_basic_iface_t ucc_cl_basic = {
    .super.super.name = "basic",
    .super.type       = UCC_CL_BASIC,
    .super.priority   = 10,
    .super.cl_lib_config =
        {
            .name   = "CL_BASIC",
            .prefix = "CL_BASIC_",
            .table  = ucc_cl_basic_lib_config_table,
            .size   = sizeof(ucc_cl_basic_lib_config_t),
        },
    .super.cl_context_config =
        {
            .name   = "CL_BASIC",
            .prefix = "CL_BASIC_",
            .table  = ucc_cl_basic_context_config_table,
            .size   = sizeof(ucc_cl_basic_context_config_t),
        },
    .super.init     = ucc_cl_basic_lib_init,
    .super.finalize = ucc_cl_basic_lib_finalize,
};
