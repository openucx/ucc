/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <sys/time.h>
#include "tl_sharp.h"

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_sharp_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_sharp_lib_config_t *tl_sharp_config =
                            ucc_derived_of(config, ucc_tl_sharp_lib_config_t);

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_lib_t, &ucc_tl_sharp.super,
                              &tl_sharp_config->super);
    memcpy(&self->cfg, tl_sharp_config, sizeof(*tl_sharp_config));
    tl_debug(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_sharp_lib_t)
{
    tl_debug(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_sharp_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_sharp_get_lib_attr(const ucc_base_lib_t *lib,
                                       ucc_base_lib_attr_t *base_attr)
{
    ucc_tl_sharp_lib_t *sharp_lib = ucc_derived_of(lib, ucc_tl_sharp_lib_t);
    ucc_tl_lib_attr_t  *attr      = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);

    attr->super.flags = 0;
    if (lib != NULL) {
        if (sharp_lib->cfg.use_internal_oob) {
            attr->super.flags |= UCC_BASE_LIB_FLAG_CTX_SERVICE_TEAM_REQUIRED;
        }
    }
    attr->super.attr.thread_mode = UCC_THREAD_MULTIPLE;
    attr->super.attr.coll_types  = UCC_TL_SHARP_SUPPORTED_COLLS;
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE) {
        if (lib == NULL) {
            return UCC_ERR_INVALID_PARAM;
        }
        attr->super.min_team_size = lib->min_team_size;
    }

    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE) {
        if (lib == NULL) {
            return UCC_ERR_INVALID_PARAM;
        }
        attr->super.max_team_size = UCC_RANK_MAX;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_get_lib_properties(ucc_base_lib_properties_t *prop)
{
    prop->default_team_size = 4;
    prop->min_team_size     = 2;
    prop->max_team_size     = UCC_RANK_MAX;
    return UCC_OK;
}
