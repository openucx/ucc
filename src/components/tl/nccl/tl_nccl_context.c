/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"

static ucc_mpool_ops_t ucc_tl_nccl_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_nccl_context_config_t *tl_nccl_config =
        ucc_derived_of(config, ucc_tl_nccl_context_config_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_nccl_config->super.tl_lib,
                              params->context);
    memcpy(&self->cfg, tl_nccl_config, sizeof(*tl_nccl_config));
    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_tl_nccl_req_mpool_ops, params->thread_mode,
                            "tl_nccl_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_nccl_req mpool");
        return status;
    }
    tl_info(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_nccl_context_t, ucc_tl_context_t);

ucc_status_t ucc_tl_nccl_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                                          ucc_base_attr_t *attr) /* NOLINT */
{
    /* TODO */
    return UCC_ERR_NOT_IMPLEMENTED;
}
