/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp.h"
#include "components/mc/base/ucc_mc_base.h"

ucc_status_t ucc_tl_sharp_get_lib_attr(const ucc_base_lib_t *lib,
                                       ucc_base_lib_attr_t *base_attr);

ucc_status_t ucc_tl_sharp_get_context_attr(const ucc_base_context_t *context,
                                           ucc_base_ctx_attr_t *base_attr);

static ucc_config_field_t ucc_tl_sharp_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_sharp_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {NULL}};

static ucc_config_field_t ucc_tl_sharp_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_sharp_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"DEVICES", "mlx5_0:1",
     "SHARP device list",
     ucc_offsetof(ucc_tl_sharp_context_config_t, dev_list),
     UCC_CONFIG_TYPE_STRING},

    {"USE_RCACHE", "y",
     "Use registration cache for sharp",
     ucc_offsetof(ucc_tl_sharp_context_config_t, use_rcache),
     UCC_CONFIG_TYPE_BOOL},

    {"REG_THRESH", "256",
     "Size threshold to register buffers",
     ucc_offsetof(ucc_tl_sharp_context_config_t, reg_threshold),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"UPROGRESS_NUM_POLLS", "9999",
     "Number of polls to do before calling user progress",
     ucc_offsetof(ucc_tl_sharp_context_config_t, uprogress_num_polls),
     UCC_CONFIG_TYPE_UINT},

    {"RAND_SEED", "0",
     "Seed for random sharp job ID. (0 - use default).",
     ucc_offsetof(ucc_tl_sharp_context_config_t, rand_seed),
     UCC_CONFIG_TYPE_UINT},
    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_sharp_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_sharp_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_sharp_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_sharp_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_sharp_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_sharp_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_sharp_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_sharp_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task);

ucc_status_t ucc_tl_sharp_team_get_scores(ucc_base_team_t   *tl_team,
                                          ucc_coll_score_t **score_p);
UCC_TL_IFACE_DECLARE(sharp, SHARP);
