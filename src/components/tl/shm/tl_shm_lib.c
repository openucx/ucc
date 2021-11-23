/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_shm_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_shm_lib_config_t *tl_shm_config =
        ucc_derived_of(config, ucc_tl_shm_lib_config_t);

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_lib_t, &ucc_tl_shm.super,
                              &tl_shm_config->super);
    memcpy(&self->cfg, tl_shm_config, sizeof(*tl_shm_config));
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_shm_lib_t)
{
    tl_info(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_shm_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_shm_get_lib_attr(const ucc_base_lib_t *lib, /* NOLINT */
                                     ucc_base_lib_attr_t  *base_attr)
{
    return UCC_OK;
}
