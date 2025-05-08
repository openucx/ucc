/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"
ucc_status_t ucc_cl_basic_get_lib_attr(const ucc_base_lib_t *lib,
                                       ucc_base_lib_attr_t  *base_attr);
ucc_status_t ucc_cl_basic_get_context_attr(const ucc_base_context_t *context,
                                           ucc_base_ctx_attr_t      *base_attr);
ucc_status_t ucc_cl_basic_mem_map(const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
                                  ucc_mem_map_memh_t *memh, ucc_mem_map_tl_t *tl_h);
ucc_status_t ucc_cl_basic_mem_unmap(const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
                                    ucc_mem_map_tl_t *tl_h);
ucc_status_t ucc_cl_basic_memh_pack(const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
                                    ucc_mem_map_tl_t *tl_h, void **packed_buffer);

ucc_status_t ucc_cl_basic_get_lib_properties(ucc_base_lib_properties_t *prop);

static ucc_config_field_t ucc_cl_basic_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_basic_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {NULL}
};

static ucs_config_field_t ucc_cl_basic_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_basic_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {NULL}
};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_basic_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_basic_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_basic_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_basic_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_basic_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_cl_basic_team_create_test(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_basic_team_destroy(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_basic_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task);

ucc_status_t ucc_cl_basic_team_get_scores(ucc_base_team_t   *cl_team,
                                          ucc_coll_score_t **score);
UCC_CL_IFACE_DECLARE(basic, BASIC);
