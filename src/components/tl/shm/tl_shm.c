/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_shm.h"
#include "components/mc/base/ucc_mc_base.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
#include "barrier/barrier.h"
#include "fanin/fanin.h"
#include "fanout/fanout.h"

ucc_status_t ucc_tl_shm_get_lib_attr(const ucc_base_lib_t *lib,
                                     ucc_base_lib_attr_t * base_attr);

ucc_status_t ucc_tl_shm_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t *     base_attr);

static const char *bcast_algs[] = {[BCAST_WW]   = "ww",
                                   [BCAST_WR]   = "wr",
                                   [BCAST_RR]   = "rr",
                                   [BCAST_RW]   = "rw",
                                   [BCAST_LAST] = NULL};

static ucc_config_field_t ucc_tl_shm_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_shm_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)},

    {"N_CONCURRENT_COLLS", "2", "Number of concurrent collective calls",
     ucc_offsetof(ucc_tl_shm_lib_config_t, n_concurrent), UCC_CONFIG_TYPE_UINT},

    {"CS", "256", "Control size of each section in shm segment",
     ucc_offsetof(ucc_tl_shm_lib_config_t, ctrl_size), UCC_CONFIG_TYPE_UINT},

    {"DS", "4096", "Data size of each section in shm segment",
     ucc_offsetof(ucc_tl_shm_lib_config_t, data_size), UCC_CONFIG_TYPE_UINT},

    {"BCAST_ALG", "wr",
     "bcast alg choice of write/read per level, can be: ww/wr/rr/rw",
     ucc_offsetof(ucc_tl_shm_lib_config_t, bcast_alg),
     UCC_CONFIG_TYPE_ENUM(bcast_algs)},

    {"BCAST_BASE_RADIX", "4", "bcast radix for base tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, bcast_base_radix),
     UCC_CONFIG_TYPE_UINT},

    {"BCAST_TOP_RADIX", "4", "bcast radix for top tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, bcast_top_radix),
     UCC_CONFIG_TYPE_UINT},

    {"REDUCE_BASE_RADIX", "4", "reduce radix for base tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, reduce_base_radix),
     UCC_CONFIG_TYPE_UINT},

    {"REDUCE_TOP_RADIX", "4", "reduce radix for top tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, reduce_top_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANIN_BASE_RADIX", "4", "fanin radix for base tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, fanin_base_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANIN_TOP_RADIX", "4", "fanin radix for top tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, fanin_top_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANOUT_BASE_RADIX", "4", "fanout radix for base tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, fanout_base_radix),
     UCC_CONFIG_TYPE_UINT},

    {"FANOUT_TOP_RADIX", "4", "fanout radix for top tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, fanout_top_radix),
     UCC_CONFIG_TYPE_UINT},

    {"BARRIER_BASE_RADIX", "4", "barrier radix for base tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, barrier_base_radix),
     UCC_CONFIG_TYPE_UINT},

    {"BARRIER_TOP_RADIX", "4", "barrier radix for top tree",
     ucc_offsetof(ucc_tl_shm_lib_config_t, barrier_top_radix),
     UCC_CONFIG_TYPE_UINT},

    {"NPOLLS", "100", "n_polls", ucc_offsetof(ucc_tl_shm_lib_config_t, n_polls),
     UCC_CONFIG_TYPE_UINT},

    {"MAX_TREES_CACHED", "8", "max num of trees that can be cached on team",
     ucc_offsetof(ucc_tl_shm_lib_config_t, max_trees_cached),
     UCC_CONFIG_TYPE_UINT},

    {"BASE_TREE_ONLY", "0",
     "if true disabling topo, "
     "forcing all processes to be on same socket/numa",
     ucc_offsetof(ucc_tl_shm_lib_config_t, base_tree_only),
     UCC_CONFIG_TYPE_BOOL},

    {"SET_PERF_PARAMS", "1",
     "changing default/user settings to optimal "
     "perf settings",
     ucc_offsetof(ucc_tl_shm_lib_config_t, set_perf_params),
     UCC_CONFIG_TYPE_BOOL},

    {"GROUP_MODE", "socket", "group mode - numa or socket",
     ucc_offsetof(ucc_tl_shm_lib_config_t, group_mode), UCC_CONFIG_TYPE_STRING},

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
                                  ucc_base_team_t *team, ucc_coll_task_t **task)
{
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_shm_bcast_init(coll_args, team, task);
    case UCC_COLL_TYPE_REDUCE:
        return ucc_tl_shm_reduce_init(coll_args, team, task);
    case UCC_COLL_TYPE_FANIN:
        return ucc_tl_shm_fanin_init(coll_args, team, task);
    case UCC_COLL_TYPE_FANOUT:
        return ucc_tl_shm_fanout_init(coll_args, team, task);
    case UCC_COLL_TYPE_BARRIER:
        return ucc_tl_shm_barrier_init(coll_args, team, task);
    default:
        break;
    }
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_tl_shm_team_get_scores(ucc_base_team_t *  tl_team,
                                        ucc_coll_score_t **score_p);
UCC_TL_IFACE_DECLARE(shm, SHM);
