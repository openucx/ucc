/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_BASIC_H_
#define UCC_CL_BASIC_H_
#include "components/cl/ucc_cl.h"
#include "components/cl/ucc_cl_log.h"
#include "components/tl/ucc_tl.h"
#include "coll_score/ucc_coll_score.h"

#ifndef UCC_CL_BASIC_DEFAULT_SCORE
#define UCC_CL_BASIC_DEFAULT_SCORE 10
#endif

typedef struct ucc_cl_basic_iface {
    ucc_cl_iface_t super;
} ucc_cl_basic_iface_t;
/* Extern iface should follow the pattern: ucc_cl_<cl_name> */
extern ucc_cl_basic_iface_t ucc_cl_basic;

typedef struct ucc_cl_basic_lib_config {
    ucc_cl_lib_config_t super;
} ucc_cl_basic_lib_config_t;

typedef struct ucc_cl_basic_context_config {
    ucc_cl_context_config_t super;
} ucc_cl_basic_context_config_t;

typedef struct ucc_cl_basic_lib {
    ucc_cl_lib_t             super;
} ucc_cl_basic_lib_t;
UCC_CLASS_DECLARE(ucc_cl_basic_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_basic_context {
    ucc_cl_context_t   super;
    ucc_tl_context_t **tl_ctxs;
    unsigned           n_tl_ctxs;
} ucc_cl_basic_context_t;
UCC_CLASS_DECLARE(ucc_cl_basic_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_basic_team {
    ucc_cl_team_t            super;
    ucc_team_multiple_req_t *team_create_req;
    ucc_tl_team_t          **tl_teams;
    unsigned                 n_tl_teams;
    ucc_coll_score_t        *score;
    ucc_score_map_t         *score_map;
} ucc_cl_basic_team_t;
UCC_CLASS_DECLARE(ucc_cl_basic_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_CL_BASIC_TEAM_CTX(_team)                                           \
    (ucc_derived_of((_team)->super.super.context, ucc_cl_basic_context_t))

#endif
