/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_cuda_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_cuda_lib_config_t *tl_config     =
        ucc_derived_of(config, ucc_tl_cuda_lib_config_t);
    size_t min_scratch_size;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_lib_t, &ucc_tl_cuda.super,
                              &tl_config->super);
    memcpy(&self->cfg, tl_config, sizeof(*tl_config));
    if (self->cfg.allgather_ring_num_chunks < 1) {
        self->cfg.allgather_ring_num_chunks = 1;
    }
    if (self->cfg.allgather_ring_num_chunks > UCC_TL_CUDA_MAX_RING_CHUNKS) {
        self->cfg.allgather_ring_num_chunks = UCC_TL_CUDA_MAX_RING_CHUNKS;
    }

    /* min scratch size should be large enough so that
     * ucc_align_down_pow2(scratch_size / nrings / nchunks / dt_size / 2, 64) > 1
     */
    min_scratch_size = 128 * 16 * ucc_dt_size(UCC_DT_FLOAT128_COMPLEX) *
                       self->cfg.allgather_ring_num_chunks;
    if (self->cfg.scratch_size < min_scratch_size) {
        self->cfg.scratch_size = min_scratch_size;
    }
    tl_debug(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_lib_t)
{
    tl_debug(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_cuda_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_cuda_get_lib_attr(const ucc_base_lib_t *lib, /* NOLINT */
                                      ucc_base_lib_attr_t  *base_attr)
{
    ucc_tl_lib_attr_t *attr      = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);

    attr->super.attr.thread_mode = UCC_THREAD_MULTIPLE;
    attr->super.attr.coll_types  = UCC_TL_CUDA_SUPPORTED_COLLS;
    attr->super.flags            = 0;
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE) {
        attr->super.min_team_size = lib->min_team_size;
    }
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE) {
        attr->super.max_team_size = UCC_TL_CUDA_MAX_PEERS;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_get_lib_properties(ucc_base_lib_properties_t *prop)
{
    prop->default_team_size = 2;
    prop->min_team_size     = 2;
    prop->max_team_size     = UCC_TL_CUDA_MAX_PEERS;
    return UCC_OK;
}
