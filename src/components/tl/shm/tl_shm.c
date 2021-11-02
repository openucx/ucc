/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "components/mc/base/ucc_mc_base.h"

ucc_status_t ucc_tl_shm_get_lib_attr(const ucc_base_lib_t *lib,
                                      ucc_base_lib_attr_t  *base_attr);

ucc_status_t ucc_tl_shm_get_context_attr(const ucc_base_context_t *context,
                                          ucc_base_ctx_attr_t      *base_attr);

static ucc_config_field_t ucc_tl_shm_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_shm_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"N_CONCURRENT_COLLS", "2", "Number of concurrent collective calls",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, n_concurrent),
     UCC_CONFIG_TYPE_UINT},

    {"CS", "128", "Control size of each section in shm segment",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, ctrl_size),
     UCC_CONFIG_TYPE_UINT},

    {"DS", "4096", "Data size of each section in shm segment",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, data_size),
     UCC_CONFIG_TYPE_UINT},

    {"GROUP_PAGE_SIZE", "4096", "Page size for group ctrls alignment", //change from cfg to detectable from system like in ucs
     ucc_offsetof(ucc_tl_ucp_lib_config_t, page_size),
     UCC_CONFIG_TYPE_UINT},

    {"BCAST_ALG", "1", "bcast alg choice of write/read per level",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, bcast_alg),
     UCC_CONFIG_TYPE_UINT},

    {"BCAST_KN_RADIX", "4", "bcast radix for knomial tree",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, bcast_kn_radix),
     UCC_CONFIG_TYPE_UINT},

    {"NPOLLS", "100", "n_polls",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, n_polls),
     UCC_CONFIG_TYPE_UINT},

    {"MAX_TREES_CACHED", "5", "max num of trees that can be cached on team",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, max_trees_cached),
     UCC_CONFIG_TYPE_UINT},

    {"GROUP_MODE", "socket", "group mode - numa or socket",
     ucc_offsetof(ucc_tl_ucp_lib_config_t, group_mode),
     UCC_CONFIG_TYPE_STRING},

    {NULL}};

static ucs_config_field_t ucc_tl_shm_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_shm_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)},

    {NULL}};

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_shm_lib_t, ucc_base_lib_t,
                          const ucc_base_lib_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_shm_lib_t, ucc_base_lib_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_shm_context_t, ucc_base_context_t,
                          const ucc_base_context_params_t *,
                          const ucc_base_config_t *);

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_shm_context_t, ucc_base_context_t);

UCC_CLASS_DEFINE_NEW_FUNC(ucc_tl_shm_team_t, ucc_base_team_t,
                          ucc_base_context_t *, const ucc_base_team_params_t *);

ucc_status_t ucc_tl_shm_team_create_test(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_shm_team_destroy(ucc_base_team_t *tl_team);

ucc_status_t ucc_tl_shm_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task);

ucc_status_t ucc_tl_shm_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p);
UCC_TL_IFACE_DECLARE(shm, SHM);
