/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_HIER_H_
#define UCC_CL_HIER_H_
#include "components/cl/ucc_cl.h"
#include "components/cl/ucc_cl_log.h"
#include "components/tl/ucc_tl.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_mpool.h"
#include "schedule/ucc_schedule_pipelined.h"

#ifdef HAVE_PROFILING_CL_HIER
#include "utils/profile/ucc_profile_on.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_CL_HIER_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_CL_HIER_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_CL_HIER_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_CL_HIER_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

#ifndef UCC_CL_HIER_DEFAULT_SCORE
#define UCC_CL_HIER_DEFAULT_SCORE 50
#endif

typedef struct ucc_cl_hier_iface {
    ucc_cl_iface_t super;
} ucc_cl_hier_iface_t;
/* Extern iface should follow the pattern: ucc_cl_<cl_name> */
extern ucc_cl_hier_iface_t ucc_cl_hier;

typedef enum {
    UCC_HIER_SBGP_NODE,
    UCC_HIER_SBGP_NODE_LEADERS,
    UCC_HIER_SBGP_NET,
    UCC_HIER_SBGP_FULL,
    UCC_HIER_SBGP_LAST,
} ucc_hier_sbgp_type_t;
//DO we need it? Potential use case: different hier sbgps over same sbgp

typedef struct ucc_cl_hier_lib_config {
    ucc_cl_lib_config_t super;
    /* List of TLs corresponding to the sbgp team,
       which are selected based on the TL scores */
    ucc_config_names_list_t sbgp_tls[UCC_HIER_SBGP_LAST];
    size_t                  a2av_node_thresh;
    ucc_pipeline_params_t   allreduce_split_rail_pipeline;
    ucc_pipeline_params_t   allreduce_rab_pipeline;
    ucc_pipeline_params_t   bcast_2step_pipeline;
    ucc_pipeline_params_t   reduce_2step_pipeline;
} ucc_cl_hier_lib_config_t;

typedef struct ucc_cl_hier_context_config {
    ucc_cl_context_config_t super;
} ucc_cl_hier_context_config_t;

typedef struct ucc_cl_hier_lib {
    ucc_cl_lib_t             super;
    ucc_cl_hier_lib_config_t cfg;
    ucc_config_allow_list_t  tls; /*< Intersection of UCC_CL_HIER_TLS
                                    vs sbgp_tls */
} ucc_cl_hier_lib_t;
UCC_CLASS_DECLARE(ucc_cl_hier_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_hier_context {
    ucc_cl_context_t super;
    ucc_mpool_t      sched_mp;
} ucc_cl_hier_context_t;
UCC_CLASS_DECLARE(ucc_cl_hier_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef enum {
    UCC_HIER_SBGP_DISABLED,
    UCC_HIER_SBGP_ENABLED
} ucc_hier_sbgp_state_t;

#define CL_HIER_MAX_SBGP_TLS 4

/* This struct represents a unit of hierarchy: a combination of
   (i) subgroup (sbgp) of processes from the original ucc_team and
   (ii) communication TL backend initialized for that subgroup.
   The backend is a superposition of at least one or several TL teams.
   Their selection is stored in score_map. */
typedef struct ucc_hier_sbgp {
    ucc_hier_sbgp_state_t state;
    ucc_sbgp_type_t       sbgp_type;
    ucc_sbgp_t           *sbgp;
    ucc_score_map_t      *score_map;
    ucc_coll_score_t     *score;
    ucc_tl_team_t        *tl_teams[CL_HIER_MAX_SBGP_TLS];
    ucc_tl_context_t     *tl_ctxs[CL_HIER_MAX_SBGP_TLS];
    int                   n_tls;
} ucc_hier_sbgp_t;

typedef struct ucc_cl_hier_team {
    ucc_cl_team_t            super;
    ucc_team_multiple_req_t *team_create_req;
    unsigned                 n_tl_teams;
    ucc_coll_score_t        *score;
    ucc_hier_sbgp_t          sbgps[UCC_HIER_SBGP_LAST];
    ucc_hier_sbgp_type_t     top_sbgp;
    /* Array of size team_size, where node_leaders[i] = the rank of i's node
       leader */
    ucc_rank_t              *node_leaders;
    /* Array of size node_leader_sbgp_size, with ranks in terms of the
       team, sorted lowest to highest. This is useful for allgatherv.
       The reason is the iterating through the node leader sbgp and map eval'ing
       the ranks can yield unsorted ranks, e.g. 2n2ppn with ranks 0 and 2 as
       leaders, leader 0 could map to rank 2 and leader 1 could map to rank 0 */
    ucc_rank_t              *leader_list;
} ucc_cl_hier_team_t;
UCC_CLASS_DECLARE(ucc_cl_hier_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_CL_HIER_SUPPORTED_COLLS                                            \
    (UCC_COLL_TYPE_ALLTOALL |                                                  \
     UCC_COLL_TYPE_ALLTOALLV |                                                 \
     UCC_COLL_TYPE_ALLGATHERV |                                                 \
     UCC_COLL_TYPE_ALLREDUCE |                                                 \
     UCC_COLL_TYPE_BARRIER |                                                   \
     UCC_COLL_TYPE_BCAST |                                                     \
     UCC_COLL_TYPE_REDUCE)

ucc_status_t ucc_cl_hier_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task);

#define UCC_CL_HIER_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_cl_hier_context_t))

#define UCC_CL_HIER_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_cl_hier_lib_t))

#define SBGP_ENABLED(_team, _sbgp)                                             \
    ((_team)->sbgps[UCC_HIER_SBGP_##_sbgp].state == UCC_HIER_SBGP_ENABLED)

#define SBGP_RANK(_team, _sbgp)                                                \
    ((_team)->sbgps[UCC_HIER_SBGP_##_sbgp].sbgp->group_rank)

#define SBGP_SIZE(_team, _sbgp)                                                \
    ((_team)->sbgps[UCC_HIER_SBGP_##_sbgp].sbgp->group_size)

#define SBGP_MAP(_team, _sbgp)                                                 \
    ((_team)->sbgps[UCC_HIER_SBGP_##_sbgp].sbgp->map)

#define SBGP_EXISTS(_team, _sbgp)                                              \
    ((NULL != (_team)->sbgps[UCC_HIER_SBGP_##_sbgp].sbgp) &&                   \
     ((_team)->sbgps[UCC_HIER_SBGP_##_sbgp].sbgp->status !=                    \
      UCC_SBGP_NOT_EXISTS))

#define SCORE_MAP(_team, _sbgp) (_team)->sbgps[UCC_HIER_SBGP_##_sbgp].score_map

#endif
