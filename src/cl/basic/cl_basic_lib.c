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
     UCS_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {NULL}
};

static ucc_status_t ucc_basic_lib_init(const ucc_lib_params_t *params,
                                       const ucc_lib_config_t *config,
                                       const ucc_cl_lib_config_t *cl_config,
                                       ucc_cl_lib_t **cl_lib)
{
    ucc_cl_basic_lib_t *lib;
    ucc_status_t        status;
    lib = (ucc_cl_basic_lib_t *)ucc_malloc(sizeof(*lib), "basic_lib");
    ucc_cl_lib_init(&lib->super, &ucc_cl_basic.super, cl_config);
    if (!lib) {
        cl_error(&lib->super, "failed to allocate %zd bytes for lib object",
                 sizeof(*lib));
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    lib->super.iface = &ucc_cl_basic.super;
    cl_info(&lib->super, "initialized lib object: %p", lib);
    *cl_lib = &lib->super;
    return UCC_OK;

error:
    return status;
}

static ucc_status_t ucc_basic_lib_finalize(ucc_cl_lib_t *cl_lib)
{
    ucc_cl_basic_lib_t *lib = ucc_derived_of(cl_lib, ucc_cl_basic_lib_t);
    cl_info(cl_lib, "finalizing lib object: %p", lib);
    free(lib);
    return UCC_OK;
}

ucc_cl_basic_iface_t ucc_cl_basic = {
    .super.super.name = "basic",
    .super.priority   = 10,
    .super.cl_lib_config =
        {
            .name   = "CL_BASIC",
            .prefix = "CL_BASIC_",
            .table  = ucc_cl_basic_lib_config_table,
            .size   = sizeof(ucc_cl_basic_lib_config_t),
        },
    .super.init     = ucc_basic_lib_init,
    .super.finalize = ucc_basic_lib_finalize,
};
