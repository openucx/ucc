/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "utils/ucc_malloc.h"

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_ucp_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_ucp_lib_config_t *tl_ucp_config =
        ucc_derived_of(config, ucc_tl_ucp_lib_config_t);
    int                            n_plugins     =
        ucc_tl_ucp.super.coll_plugins.n_components;
    ucc_tl_coll_plugin_iface_t *tlcp;
    int                         i;
    ucc_status_t                status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_lib_t, &ucc_tl_ucp.super,
                              &tl_ucp_config->super);
    memcpy(&self->cfg, tl_ucp_config, sizeof(*tl_ucp_config));
    if (tl_ucp_config->kn_radix > 0) {
        self->cfg.barrier_kn_radix        = tl_ucp_config->kn_radix;
        self->cfg.allreduce_kn_radix      = tl_ucp_config->kn_radix;
        self->cfg.allreduce_sra_kn_radix  = tl_ucp_config->kn_radix;
        self->cfg.reduce_scatter_kn_radix = tl_ucp_config->kn_radix;
        self->cfg.allgather_kn_radix      = tl_ucp_config->kn_radix;
        self->cfg.bcast_kn_radix          = tl_ucp_config->kn_radix;
        self->cfg.bcast_sag_kn_radix      = tl_ucp_config->kn_radix;
        self->cfg.reduce_kn_radix         = tl_ucp_config->kn_radix;
        self->cfg.scatter_kn_radix        = tl_ucp_config->kn_radix;
    }

    self->tlcp_configs = NULL;
    if (n_plugins) {
        self->tlcp_configs = ucc_malloc(sizeof(void*)*n_plugins, "tlcp_configs");
        if (!self->tlcp_configs) {
            tl_error(&self->super, "failed to allocate %zd bytes for tlcp_configs",
                     sizeof(void*)*n_plugins);
            status = UCC_ERR_NO_MEMORY;
            goto err;
        }
        for (i = 0; i < n_plugins; i++) {
            tlcp = ucc_derived_of(ucc_tl_ucp.super.coll_plugins.components[i],
                                  ucc_tl_coll_plugin_iface_t);
            tlcp->id = i;
            self->tlcp_configs[i] = ucc_malloc(tlcp->config.size, "tlcp_cfg");
            if (!self->tlcp_configs[i]) {
                tl_error(&self->super, "failed to allocate %zd bytes for tlcp_cfg",
                         tlcp->config.size);
                status = UCC_ERR_NO_MEMORY;
                goto err_cfg;
            }
            status = ucc_config_parser_fill_opts(self->tlcp_configs[i],
                                                 tlcp->config.table,
                                                 params->full_prefix,
                                                 ucc_tl_ucp.super.tl_lib_config.prefix, 0);
            if (status != UCC_OK) {
                tl_error(&self->super, "failed to read tlcp config");
                goto err_cfg;
            }
        }
    }
    tl_info(&self->super, "initialized lib object: %p", self);
    return UCC_OK;

err_cfg:
    for (i--; i >= 0; i--) {
        ucc_free(self->tlcp_configs[i]);
    }
err:
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_lib_t)
{
    tl_info(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_ucp_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_ucp_get_lib_attr(const ucc_base_lib_t *lib, /* NOLINT */
                                     ucc_base_lib_attr_t  *base_attr)
{
    ucc_tl_lib_attr_t *attr = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);
    ucs_status_t   status;
    ucp_lib_attr_t params;
    memset(&params, 0, sizeof(ucp_lib_attr_t));
    params.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
    status            = ucp_lib_query(&params);
    if (status != UCS_OK) {
        ucc_error("failed to query UCP lib attributes");
        return ucs_status_to_ucc_status(status);
    }
    switch (params.max_thread_level)
    {
        case UCS_THREAD_MODE_SINGLE:
            attr->super.attr.thread_mode = UCC_THREAD_SINGLE;
            break;

        case UCS_THREAD_MODE_SERIALIZED:
            attr->super.attr.thread_mode = UCC_THREAD_SINGLE;
            break;

        case UCS_THREAD_MODE_MULTI:
            attr->super.attr.thread_mode = UCC_THREAD_MULTIPLE;
            break;

        default:
            ucc_error("Unsupported UCS thread mode");
            return UCC_ERR_NO_RESOURCE;
    }
    attr->super.attr.coll_types = UCC_TL_UCP_SUPPORTED_COLLS;
    attr->super.flags           = UCC_BASE_LIB_FLAG_TEAM_ID_REQUIRED;
    return UCC_OK;
}
