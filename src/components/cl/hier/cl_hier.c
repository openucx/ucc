/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "utils/ucc_malloc.h"
ucc_status_t ucc_cl_hier_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t  *base_attr);
ucc_status_t ucc_cl_hier_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_cl_hier_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_hier_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_lib_config_table)},

    {"NODE_SBGP_TLS", "ucp", "TLS to be used for NODE subgroup",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_NODE]),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {"NET_SBGP_TLS", "ucp", "TLS to be used for NET subgroup",
     ucc_offsetof(ucc_cl_hier_lib_config_t, sbgp_tls[UCC_HIER_SBGP_NET]),
     UCC_CONFIG_TYPE_STRING_ARRAY},

    {NULL}};

static ucs_config_field_t ucc_cl_hier_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_cl_hier_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_cl_context_config_table)},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_hier_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_cl_hier_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_cl_hier_team_create_test(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_hier_team_destroy(ucc_base_team_t *cl_team);

ucc_status_t ucc_cl_hier_team_get_scores(ucc_base_team_t   *cl_team,
                                         ucc_coll_score_t **score);
UCC_CL_IFACE_DECLARE(hier, HIER);
