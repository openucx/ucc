/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_cuda_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_cuda_lib_config_t *tl_config =
        ucc_derived_of(config, ucc_tl_cuda_lib_config_t);
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_lib_t, &ucc_tl_cuda.super,
                              &tl_config->super);
    memcpy(&self->cfg, tl_config, sizeof(*tl_config));
    if (self->cfg.allgather_ring_num_chunks < 1) {
        self->cfg.allgather_ring_num_chunks = 1;
    }
    if (self->cfg.allgather_ring_num_chunks > UCC_TL_CUDA_MAX_RING_CHUNKS) {
        self->cfg.allgather_ring_num_chunks = UCC_TL_CUDA_MAX_RING_CHUNKS;
    }
    tl_info(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_lib_t)
{
    tl_info(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_cuda_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_cuda_get_lib_attr(const ucc_base_lib_t *lib, /* NOLINT */
                                      ucc_base_lib_attr_t  *base_attr)
{
    ucc_tl_lib_attr_t *attr      = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);

    attr->super.flags            = 0;
    attr->super.attr.thread_mode = UCC_THREAD_MULTIPLE;
    attr->super.attr.coll_types  = UCC_TL_CUDA_SUPPORTED_COLLS;
    return UCC_OK;
}
