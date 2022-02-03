/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_ec_base.h"

ucc_config_field_t ucc_ec_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC logging level. Messages with a level higher or equal to the "
     "selected will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_ec_config_t, log_component), UCC_CONFIG_TYPE_LOG_COMP},

    {NULL}};
