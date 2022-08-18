/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "components/mc/base/ucc_mc_base.h"
#include "allgatherv/allgatherv.h"

ucc_status_t ucc_tl_nccl_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t  *base_attr);

ucc_status_t ucc_tl_nccl_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_tl_nccl_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_nccl_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {NULL}};

const char* ucc_tl_nccl_completion_sync_names[] = {
    [UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT]  = "event",
    [UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS] = "driver",
    [UCC_TL_NCCL_COMPLETION_SYNC_TYPE_AUTO]   = "auto",
    [UCC_TL_NCCL_COMPLETION_SYNC_TYPE_LAST]   = NULL
};

static ucs_config_field_t ucc_tl_nccl_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_nccl_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"SYNC", "auto",
     "Determines how UCC tests completion of NCCL collective",
     ucs_offsetof(ucc_tl_nccl_context_config_t, sync_type),
     UCS_CONFIG_TYPE_ENUM(ucc_tl_nccl_completion_sync_names)
    },

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_nccl_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_nccl_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_nccl_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_nccl_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_nccl_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_nccl_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_nccl_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_nccl_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_nccl_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);
UCC_TL_IFACE_DECLARE(nccl, NCCL);

__attribute__((constructor)) static void tl_nccl_iface_init(void)
{
    ucc_tl_nccl.super.alg_info[ucc_ilog2(UCC_COLL_TYPE_ALLGATHERV)] =
        ucc_tl_nccl_allgatherv_algs;
}
