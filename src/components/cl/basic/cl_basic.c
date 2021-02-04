/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_malloc.h"
ucc_status_t ucc_cl_basic_get_lib_attr(const ucc_base_lib_t *lib, ucc_base_attr_t *base_attr);
ucc_status_t ucc_cl_basic_get_context_attr(const ucc_base_context_t *context, ucc_base_attr_t *base_attr);

static ucc_config_field_t ucc_cl_basic_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_basic_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {NULL}
};

static ucs_config_field_t ucc_cl_basic_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_basic_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {"TEST_PARAM", "5", "For dbg test purpuse : don't commit",
     ucc_offsetof(ucc_cl_basic_context_config_t, test_param),
     UCC_CONFIG_TYPE_UINT},

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
ucc_status_t ucc_cl_basic_coll_init(ucc_base_coll_op_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task);
UCC_CL_IFACE_DECLARE(basic, BASIC);
