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
