/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucc_global_opts.h"
#include "api/ucc.h"
#include "api/ucc_status.h"

ucc_status_t ucc_init_version(unsigned api_major_version,
                              unsigned api_minor_version,
                              const ucc_lib_params_t *params,
                              const ucc_lib_config_h *config,
                              ucc_lib_h *lib_p)
{
    return UCC_ERR_NOT_IMPLEMENTED;
}
