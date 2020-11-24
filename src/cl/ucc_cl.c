/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_cl.h"

ucc_config_field_t ucc_cl_lib_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC CL logging level. Messages with a level higher or equal to the "
     "selected will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_cl_lib_config_t, log_component),
     UCC_CONFIG_TYPE_LOG_COMP},

    {"PRIORITY", "-1",
     "UCC CL priority.\n"
     "Possible values are: [1,inf]",
     ucc_offsetof(ucc_cl_lib_config_t, priority), UCC_CONFIG_TYPE_INT},

    {NULL}
};

const char *ucc_cl_names[] = {
    [UCC_CL_BASIC] = "basic",
    [UCC_CL_ALL]   = "all",
    [UCC_CL_LAST]  = NULL
};

UCC_CLASS_INIT_FUNC(ucc_cl_lib_t, ucc_cl_iface_t *cl_iface,
                    const ucc_lib_config_t *config,
                    const ucc_cl_lib_config_t *cl_config)
{
    self->iface         = cl_iface;
    self->log_component = cl_config->log_component;
    ucc_strncpy_safe(self->log_component.name, cl_iface->cl_lib_config.name,
                     sizeof(self->log_component.name));
    self->priority =
        (-1 == cl_config->priority) ? cl_iface->priority : cl_config->priority;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_lib_t)
{
}

UCC_CLASS_DEFINE(ucc_cl_lib_t, void);
