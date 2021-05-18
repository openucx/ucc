/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"
#include "components/tl/ucc_tl.h"
#include "core/ucc_global_opts.h"
#include "utils/ucc_math.h"

/* NOLINTNEXTLINE  TODO params is not used*/
UCC_CLASS_INIT_FUNC(ucc_cl_basic_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_lib_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_lib_config_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_lib_t, &ucc_cl_basic.super, cl_config);
    cl_info(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_basic_lib_t)
{
    cl_info(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_cl_basic_lib_t, ucc_cl_lib_t);

ucc_status_t ucc_cl_basic_get_lib_attr(const ucc_base_lib_t *lib,
                                       ucc_base_attr_t *base_attr)
{
    ucc_cl_lib_attr_t     *attr   = ucc_derived_of(base_attr, ucc_cl_lib_attr_t);
    ucc_cl_lib_t          *cl_lib = ucc_derived_of(lib, ucc_cl_lib_t);
    ucc_status_t           status;
    ucc_tl_lib_attr_t      ucp_tl_attr, nccl_tl_attr;
    ucc_component_iface_t *ucp_iface, *nccl_iface;
    ucc_tl_iface_t        *tl_ucp_iface, *tl_nccl_iface;
    attr->tls = &cl_lib->tls;
    ucp_iface = ucc_get_component(&ucc_global_config.tl_framework, "ucp");
    if (!ucp_iface) {
    	cl_error(lib, "failed to get UCP component");
    	return UCC_ERR_NO_RESOURCE;
    }
    tl_ucp_iface = ucc_derived_of(ucp_iface, ucc_tl_iface_t);
    memset(&ucp_tl_attr, 0, sizeof(ucc_tl_lib_attr_t));
    status = tl_ucp_iface->lib.get_attr(NULL, &ucp_tl_attr.super);
    if (UCC_OK != status) {
        cl_error(lib, "failed to query cl lib attributes");
        return status;
    }
    attr->super.attr.thread_mode = ucp_tl_attr.super.attr.thread_mode;
    attr->super.attr.coll_types  = ucp_tl_attr.super.attr.coll_types;
    nccl_iface = ucc_get_component(&ucc_global_config.tl_framework, "nccl");
    if (nccl_iface) {
        tl_nccl_iface = ucc_derived_of(nccl_iface, ucc_tl_iface_t);
        memset(&nccl_tl_attr, 0, sizeof(ucc_tl_lib_attr_t));
        status = tl_nccl_iface->lib.get_attr(NULL, &nccl_tl_attr.super);
        if (UCC_OK != status) {
            cl_error(lib, "failed to query cl lib attributes");
            return status;
        }
        attr->super.attr.thread_mode = ucc_min(attr->super.attr.thread_mode,
    			nccl_tl_attr.super.attr.thread_mode);
        attr->super.attr.coll_types  |= nccl_tl_attr.super.attr.coll_types;
    }

    return UCC_OK;
}
