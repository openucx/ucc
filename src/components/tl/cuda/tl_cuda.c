/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "core/ucc_team.h"
#include "components/mc/base/ucc_mc_base.h"

static ucc_config_field_t ucc_tl_cuda_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"MAX_CONCURRENT", "8",
     "Maximum number of outstanding colls",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, max_concurrent),
     UCC_CONFIG_TYPE_UINT},

    {"SCRATCH_SIZE", "1Mb",
     "Size of the internal scratch buffer",
     ucc_offsetof(ucc_tl_cuda_lib_config_t, scratch_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_lib_t, ucc_base_lib_t);

ucc_status_t ucc_tl_cuda_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t *base_attr);

static ucs_config_field_t ucc_tl_cuda_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_cuda_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {NULL}};

ucc_status_t ucc_tl_cuda_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t *base_attr);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_cuda_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_cuda_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_cuda_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);

UCC_TL_IFACE_DECLARE(cuda, CUDA);
