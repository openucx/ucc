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
    }
    tl_info(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
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
    return UCC_OK;
}
