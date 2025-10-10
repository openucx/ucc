/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#ifdef HAVE_NVLS
#include "tl_cuda_nvls.h"
#endif

/* NOLINTNEXTLINE  params is not used*/
UCC_CLASS_INIT_FUNC(ucc_tl_cuda_lib_t, const ucc_base_lib_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_tl_cuda_lib_config_t *tl_config =
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
#ifdef HAVE_NVLS
    if (self->cfg.nvls_sm_count < 1) {
        tl_error(
            &self->super,
            "nvls_sm_count is too small, min is 1, please check NVLS_SM_COUNT "
            "config parameter");
        return UCC_ERR_INVALID_PARAM;
    }
    if (self->cfg.nvls_sm_count > UCC_TL_CUDA_MAX_NVLS_SM_COUNT) {
        tl_error(
            &self->super,
            "nvls_sm_count is too large, max is %d, please check NVLS_SM_COUNT "
            "config parameter",
            UCC_TL_CUDA_MAX_NVLS_SM_COUNT);
        return UCC_ERR_INVALID_PARAM;
    }
    if (self->cfg.nvls_threads < 1) {
        tl_error(
            &self->super,
            "nvls_threads is too small, min is 1, please check NVLS_THREADS "
            "config parameter");
        return UCC_ERR_INVALID_PARAM;
    }
    if (self->cfg.nvls_threads > UCC_TL_CUDA_MAX_NVLS_THREADS) {
        tl_error(
            &self->super,
            "nvls_threads is too large, max is %d, please check NVLS_THREADS "
            "config parameter",
            UCC_TL_CUDA_MAX_NVLS_THREADS);
        return UCC_ERR_INVALID_PARAM;
    }
#endif

    /* min scratch size should be large enough so that
     * ucc_align_down_pow2(scratch_size / nrings / nchunks / dt_size / 2, 64) > 1
     */
    min_scratch_size = 128 * 16 * ucc_dt_size(UCC_DT_FLOAT128_COMPLEX) *
                       self->cfg.allgather_ring_num_chunks;
    if (self->cfg.scratch_size < min_scratch_size) {
        self->cfg.scratch_size = min_scratch_size;
    }

    /* Initialize topology pointer to NULL - will be lazily initialized */
    self->topo = NULL;

    tl_debug(&self->super, "initialized lib object: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_lib_t)
{
    if (self->topo) {
        ucc_tl_cuda_topo_destroy(self->topo);
    }
    tl_debug(&self->super, "finalizing lib object: %p", self);
}

UCC_CLASS_DEFINE(ucc_tl_cuda_lib_t, ucc_tl_lib_t);

ucc_status_t ucc_tl_cuda_get_lib_attr(const ucc_base_lib_t *lib, /* NOLINT */
                                      ucc_base_lib_attr_t  *base_attr)
{
    ucc_tl_lib_attr_t *attr      = ucc_derived_of(base_attr, ucc_tl_lib_attr_t);

    attr->super.attr.thread_mode = UCC_THREAD_MULTIPLE;
    attr->super.attr.coll_types  = UCC_TL_CUDA_SUPPORTED_COLLS;
#ifdef HAVE_NVLS
    attr->super.attr.coll_types |= UCC_COLL_TYPE_ALLREDUCE;
#endif
    attr->super.flags = 0;
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE) {
        attr->super.min_team_size = lib->min_team_size;
    }
    if (base_attr->mask & UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE) {
#ifdef HAVE_NVLS
        /* Advertise maximum possible team size. Actual NVLS support
         * will be checked during team creation. */
        attr->super.max_team_size = UCC_TL_CUDA_MAX_NVLS_PEERS;
#else
        attr->super.max_team_size = UCC_TL_CUDA_MAX_PEERS;
#endif
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_get_lib_properties(ucc_base_lib_properties_t *prop)
{
    prop->default_team_size = 2;
    prop->min_team_size     = 2;
#ifdef HAVE_NVLS
    prop->max_team_size = UCC_TL_CUDA_MAX_NVLS_PEERS;
#else
    prop->max_team_size = UCC_TL_CUDA_MAX_PEERS;
#endif
    return UCC_OK;
}
