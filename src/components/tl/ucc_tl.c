/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_tl.h"
#include "utils/ucc_log.h"

ucc_config_field_t ucc_tl_lib_config_table[] = {
    {NULL}
};

ucc_config_field_t ucc_tl_context_config_table[] = {
    {NULL}
};

UCC_CLASS_INIT_FUNC(ucc_tl_lib_t, ucc_tl_iface_t *tl_iface,
                    const ucc_tl_lib_config_t *tl_config)
{
    self->iface         = tl_iface;
    self->super.log_component = tl_config->super.log_component;
    ucc_strncpy_safe(self->super.log_component.name,
                     tl_iface->tl_lib_config.name,
                     sizeof(self->super.log_component.name));
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_lib_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_lib_t, void);

UCC_CLASS_INIT_FUNC(ucc_tl_context_t, ucc_tl_lib_t *tl_lib)
{
    self->super.lib = &tl_lib->super;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_context_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_context_t, void);

ucc_status_t ucc_tl_context_config_read(ucc_tl_lib_t *tl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_tl_context_config_t **tl_config)
{
    ucc_status_t status;
    status = ucc_base_config_read(config->lib->full_prefix,
                                  &tl_lib->iface->tl_context_config,
                                  (ucc_base_config_t **)tl_config);
    if (UCC_OK == status) {
        (*tl_config)->tl_lib = tl_lib;
    }
    return status;
}

ucc_status_t ucc_tl_lib_config_read(ucc_tl_iface_t *iface,
                                    const char *full_prefix,
                                    const ucc_lib_config_t *config,
                                    ucc_tl_lib_config_t **tl_config)
{
    return ucc_base_config_read(full_prefix, &iface->tl_lib_config,
                                (ucc_base_config_t **)tl_config);
}
