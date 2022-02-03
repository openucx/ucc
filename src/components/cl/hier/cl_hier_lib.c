/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "utils/ucc_malloc.h"
#include "components/tl/ucc_tl.h"
#include "core/ucc_global_opts.h"
#include "utils/ucc_math.h"

/* NOLINTNEXTLINE  TODO params is not used*/
UCC_CLASS_INIT_FUNC(ucc_cl_hier_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_hier_lib_config_t *cl_hier_config =
        ucc_derived_of(config, ucc_cl_hier_lib_config_t);
    ucc_config_names_array_t        requested_sbgp_tls;
    ucc_status_t                    status;
    int                             i, all_tls_requested;

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_lib_t, &ucc_cl_hier.super,
                              &cl_hier_config->super);
    memcpy(&self->cfg, cl_hier_config, sizeof(*cl_hier_config));

    requested_sbgp_tls.count = 0;
    all_tls_requested   = 0;
    for (i = 0; i < UCC_HIER_SBGP_LAST; i++) {
        status = ucc_config_names_array_dup(&self->cfg.sbgp_tls[i],
                                            &cl_hier_config->sbgp_tls[i]);
        if (UCC_OK != status) {
            cl_error(&self->super, "failed to dup sbgp tls array");
            return status;
        }
        if (!ucc_config_names_array_is_all(&cl_hier_config->sbgp_tls[i])) {
            status = ucc_config_names_array_merge(&requested_sbgp_tls,
                                                  &cl_hier_config->sbgp_tls[i]);
            if (ucc_unlikely(UCC_OK != status)) {
                cl_error(&self->super, "failed to merge tls config names arrays");
                return status;
            }
        } else {
            all_tls_requested = 1;
        }
    }

    if (requested_sbgp_tls.count == 0 || all_tls_requested) {
        /* Some sbgp tls list contained "all". This means we should use
           all possible TLs available to this CL: cl_hier_config->super.tls */
        status = ucc_config_names_array_dup(&self->tls,
                                            &cl_hier_config->super.tls);
    } else {
        status = ucc_config_names_array_dup(&self->tls,
                                            &requested_sbgp_tls);
    }
    if (requested_sbgp_tls.count > 0) {
        ucc_config_names_array_free(&requested_sbgp_tls);
    }
    cl_info(&self->super, "initialized lib object: %p", self);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_hier_lib_t)
{
    int i;

    cl_info(&self->super, "finalizing lib object: %p", self);
    for (i = 0; i < UCC_HIER_SBGP_LAST; i++) {
        ucc_config_names_array_free(&self->cfg.sbgp_tls[i]);
    }
    ucc_config_names_array_free(&self->tls);
}

UCC_CLASS_DEFINE(ucc_cl_hier_lib_t, ucc_cl_lib_t);

static inline ucc_status_t check_tl_lib_attr(const ucc_base_lib_t *lib,
                                             ucc_tl_iface_t *      tl_iface,
                                             ucc_cl_lib_attr_t *   attr)
{
    ucc_tl_lib_attr_t tl_attr;
    ucc_status_t      status;

    memset(&tl_attr, 0, sizeof(tl_attr));
    status = tl_iface->lib.get_attr(NULL, &tl_attr.super);
    if (UCC_OK != status) {
        cl_error(lib, "failed to query tl %s lib attributes",
                 tl_iface->super.name);
        return status;
    }
    attr->super.attr.thread_mode =
        ucc_min(attr->super.attr.thread_mode, tl_attr.super.attr.thread_mode);
    attr->super.flags |= tl_attr.super.flags;
    return UCC_OK;
}

ucc_status_t ucc_cl_hier_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t  *base_attr)
{
    ucc_cl_lib_attr_t *attr   = ucc_derived_of(base_attr, ucc_cl_lib_attr_t);
    ucc_cl_hier_lib_t *cl_lib = ucc_derived_of(lib, ucc_cl_hier_lib_t);
    ucc_config_names_array_t *tls = &cl_lib->tls;
    ucc_tl_iface_t           *tl_iface;
    int                       i;
    ucc_status_t              status;

    attr->tls                    = &cl_lib->tls;
    attr->super.attr.thread_mode = UCC_THREAD_MULTIPLE;
    attr->super.attr.coll_types  = UCC_CL_HIER_SUPPORTED_COLLS;
    attr->super.flags            = 0;
    if (tls->count == 1 && !strcmp(tls->names[0], "all")) {
        /* Check all available components, since CL_HIER_TLS == "all" */
        for (i = 0; i < ucc_global_config.tl_framework.n_components; i++) {
            tl_iface = ucc_derived_of(
                ucc_global_config.tl_framework.components[i], ucc_tl_iface_t);
            ucc_assert(tl_iface);
            if (UCC_OK != (status = check_tl_lib_attr(lib, tl_iface, attr))) {
                return status;
            }
        }
    } else {
        for (i = 0; i < tls->count; i++) {
            /* Check TLs proveded in CL_HIER_TLS. Not all of them could be
               available, check for NULL. */
            tl_iface = ucc_derived_of(
                ucc_get_component(&ucc_global_config.tl_framework,
                                  tls->names[i]),
                ucc_tl_iface_t);
            if (!tl_iface) {
                cl_warn(lib, "tl %s is not available", tls->names[i]);
                continue;
            }
            if (UCC_OK != (status = check_tl_lib_attr(lib, tl_iface, attr))) {
                return status;
            }
        }
    }
    return UCC_OK;
}
