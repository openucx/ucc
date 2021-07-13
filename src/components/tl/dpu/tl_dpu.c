/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_dpu.h"

ucc_status_t ucc_tl_dpu_get_lib_attr(const ucc_base_lib_t *lib,
                                     ucc_base_lib_attr_t  *base_attr);
ucc_status_t ucc_tl_dpu_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_tl_dpu_lib_config_table[] = {
    {   "",
        "",
        NULL,
        ucc_offsetof(ucc_tl_dpu_lib_config_t, super),
        UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)
    },
    {NULL}};

static ucs_config_field_t ucc_tl_dpu_context_config_table[] = {
    {   "",
        "",
        NULL,
        ucc_offsetof(ucc_tl_dpu_context_config_t, super),
        UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)
    },

    {"SERVER_HOSTNAME", "",
     "Bluefield IP address",
     ucc_offsetof(ucc_tl_dpu_context_config_t, server_hname),
     UCC_CONFIG_TYPE_STRING
    },

    {"SERVER_PORT", "13337",
     "Bluefield DPU port",
     ucc_offsetof(ucc_tl_dpu_context_config_t, server_port),
     UCC_CONFIG_TYPE_UINT
    },

    {"ENABLE", "0",
     "Assume server is running on BF",
     ucc_offsetof(ucc_tl_dpu_context_config_t, use_dpu),
     UCC_CONFIG_TYPE_UINT
    },

    {"HOST_DPU_LIST", "",
     "A host-dpu list used to identify the DPU IP",
     ucc_offsetof(ucc_tl_dpu_context_config_t, host_dpu_list),
     UCC_CONFIG_TYPE_STRING
    },
    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_dpu_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_dpu_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_dpu_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_dpu_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_dpu_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_dpu_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_dpu_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_dpu_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_dpu_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);

UCC_TL_IFACE_DECLARE(dpu, DPU);