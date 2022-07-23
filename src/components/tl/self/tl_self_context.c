/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "tl_self.h"
#include "utils/arch/cpu.h"
#include <limits.h>

UCC_CLASS_INIT_FUNC(ucc_tl_self_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t         *config)
{
    ucc_tl_self_context_config_t *tl_self_config =
        ucc_derived_of(config, ucc_tl_self_context_config_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_self_config->super,
                              params->context);
    memcpy(&self->cfg, tl_self_config, sizeof(*tl_self_config));

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_self_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX, NULL,
                            params->thread_mode, "tl_self_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_self_req mpool");
        return status;
    }

    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_self_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_self_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_self_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr /* NOLINT */)
{
    return UCC_OK;
}
