/**
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_DOCA_UROM_H_
#define UCC_CL_DOCA_UROM_H_

#include "components/cl/ucc_cl.h"
#include "components/cl/ucc_cl_log.h"
#include "components/tl/ucc_tl.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_mpool.h"

#include <doca_dev.h>
#include <doca_urom.h>
#include <doca_pe.h>
#include <doca_ctx.h>
#include <doca_buf.h>

#include "cl_doca_urom_common.h"
#include "cl_doca_urom_worker_ucc.h"

#include <urom_ucc.h>

#ifndef UCC_CL_DOCA_UROM_DEFAULT_SCORE
#define UCC_CL_DOCA_UROM_DEFAULT_SCORE 100
#endif

#define UCC_CL_DOCA_UROM_ADDR_MAX_LEN  1024
#define UCC_CL_DOCA_UROM_MAX_TEAMS     16

typedef struct ucc_cl_doca_urom_iface {
    ucc_cl_iface_t super;
} ucc_cl_doca_urom_iface_t;

// Extern iface should follow the pattern: ucc_cl_<cl_name>
extern ucc_cl_doca_urom_iface_t ucc_cl_doca_urom;

typedef struct ucc_cl_doca_urom_lib_config {
    ucc_cl_lib_config_t super;
} ucc_cl_doca_urom_lib_config_t;

typedef struct ucc_cl_doca_urom_context_config {
    ucc_cl_context_config_t  super;
    ucs_config_names_array_t plugin_envs;
    char                    *device;
    char                    *plugin_name;
    int                      doca_log_level;
} ucc_cl_doca_urom_context_config_t;

typedef struct ucc_cl_doca_urom_ctx {
    struct doca_urom_service                   *urom_service;
    struct doca_urom_worker                    *urom_worker;
    struct doca_urom_domain                    *urom_domain;
    struct doca_pe                             *urom_pe;
    const struct doca_urom_service_plugin_info *ucc_info;
    void                                       *urom_worker_addr;
    size_t                                      urom_worker_len;
    uint64_t                                    worker_id;
    void                                       *urom_ucc_context;
    ucc_rank_t                                  ctx_rank;
    struct doca_dev                            *dev;
} ucc_cl_doca_urom_ctx_t;

typedef struct ucc_cl_doca_urom_lib {
    ucc_cl_lib_t                  super;
    ucc_cl_doca_urom_lib_config_t cfg;
    int                           tl_ucp_index;
} ucc_cl_doca_urom_lib_t;
UCC_CLASS_DECLARE(ucc_cl_doca_urom_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_doca_urom_context {
    ucc_cl_context_t                  super;
    ucc_mpool_t                       sched_mp;
    ucc_cl_doca_urom_ctx_t            urom_ctx;
    ucc_cl_doca_urom_context_config_t cfg;
} ucc_cl_doca_urom_context_t;
UCC_CLASS_DECLARE(ucc_cl_doca_urom_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_doca_urom_team {
    ucc_cl_team_t                  super;
    ucc_team_h                   **teams;
    unsigned                       n_teams;
    ucc_coll_score_t              *score;
    ucc_score_map_t               *score_map;
    struct ucc_cl_doca_urom_result res; // used for the cookie
} ucc_cl_doca_urom_team_t;
UCC_CLASS_DECLARE(ucc_cl_doca_urom_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

ucc_status_t ucc_cl_doca_urom_coll_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task);

#define UCC_CL_DOCA_UROM_TEAM_CTX(_team)                                       \
    (ucc_derived_of((_team)->super.super.context, ucc_cl_doca_urom_context_t))

#endif
