/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"

ucc_status_t ucc_tl_mlx5_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t * base_attr);
ucc_status_t ucc_tl_mlx5_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t *     base_attr);

static ucc_config_field_t ucc_tl_mlx5_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_mlx5_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"DM_BUF_SIZE", "8k", "Size of the pre-allocated DeviceMemory buffer",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dm_buf_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"DM_BUF_NUM", "auto", "Number of DM buffers to alloc",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dm_buf_num),
     UCC_CONFIG_TYPE_ULUNITS},

    {"BLOCK_SIZE", "0",
     "Size of the blocks that are sent using blocked AlltoAll Algorithm",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, block_size), UCC_CONFIG_TYPE_UINT},

    {"NUM_DCI_QPS", "16",
     "Number of parallel DCI QPs that will be used per team",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, num_dci_qps), UCC_CONFIG_TYPE_UINT},

    {"DC_THRESHOLD", "128",
     "If number of nodes >= DC_THRESHOLD then DC QPs "
     "are used instead of RC",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dc_threshold),
     UCC_CONFIG_TYPE_UINT},

    {"DM_HOST", "n",
     "Use host registered memory instead of DM for "
     "transpose staging",
     ucc_offsetof(ucc_tl_mlx5_lib_config_t, dm_host), UCC_CONFIG_TYPE_BOOL},

    {NULL}};

static ucc_config_field_t ucc_tl_mlx5_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_mlx5_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {"NET_DEVICES", "", "Specifies which network device(s) to use",
     ucc_offsetof(ucc_tl_mlx5_context_config_t, devices),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mlx5_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mlx5_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_mlx5_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_mlx5_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_mlx5_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_mlx5_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task);

ucc_status_t ucc_tl_mlx5_team_get_scores(ucc_base_team_t *  tl_team,
                                         ucc_coll_score_t **score);

UCC_TL_IFACE_DECLARE(mlx5, MLX5);

ucc_status_t ucc_tl_mlx5_context_create_epilog(ucc_base_context_t *context);

__attribute__((constructor)) static void tl_mlx5_iface_init(void)
{
    ucc_tl_mlx5.super.context.create_epilog = ucc_tl_mlx5_context_create_epilog;
}
