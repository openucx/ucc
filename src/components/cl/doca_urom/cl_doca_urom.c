/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_doca_urom.h"
#include "utils/ucc_malloc.h"

ucc_status_t ucc_cl_doca_urom_get_lib_attr(const ucc_base_lib_t *lib,
                                           ucc_base_lib_attr_t  *base_attr);

ucc_status_t ucc_cl_doca_urom_get_context_attr(const ucc_base_context_t *context,
                                               ucc_base_ctx_attr_t *base_attr);

ucc_status_t ucc_cl_doca_urom_mem_map(const ucc_base_context_t *context,
                                      int type, void *memh, void *tl_h);

ucc_status_t ucc_cl_doca_urom_mem_unmap(const ucc_base_context_t *context,
                                        int type, void *tl_h);

ucc_status_t ucc_cl_doca_urom_memh_pack(const ucc_base_context_t *context,
                                        int type, void *memh, void **packed_buffer);

ucc_status_t ucc_cl_doca_urom_get_lib_properties(ucc_base_lib_properties_t *prop);

static ucc_config_field_t ucc_cl_doca_urom_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_doca_urom_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {NULL}};

static ucs_config_field_t ucc_cl_doca_urom_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_doca_urom_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {"PLUGIN_ENVS", "",
     "Comma separated envs to pass to the worker plugin",
     ucc_offsetof(ucc_cl_doca_urom_context_config_t, plugin_envs),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {"DEVICE", "mlx5_0",
     "DPU device",
     ucc_offsetof(ucc_cl_doca_urom_context_config_t, device),
     UCC_CONFIG_TYPE_STRING},

    {"PLUGIN_NAME", "libucc_doca_urom_plugin",
     "Name of plugin library",
     ucc_offsetof(ucc_cl_doca_urom_context_config_t, plugin_name),
     UCC_CONFIG_TYPE_STRING},

    {"DOCA_LOG_LEVEL", "10",
     "DOCA log level",
     ucc_offsetof(ucc_cl_doca_urom_context_config_t, doca_log_level),
     UCC_CONFIG_TYPE_INT},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_doca_urom_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_doca_urom_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_doca_urom_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_doca_urom_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_doca_urom_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_cl_doca_urom_team_create_test(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_doca_urom_team_destroy(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_doca_urom_coll_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task);

ucc_status_t ucc_cl_doca_urom_team_get_scores(ucc_base_team_t   *cl_team,
                                              ucc_coll_score_t **score);

UCC_CL_IFACE_DECLARE(doca_urom, DOCA_UROM);
