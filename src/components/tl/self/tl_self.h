/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SELF_H_
#define UCC_TL_SELF_H_
#include <ucs/memory/memory_type.h>
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "core/ucc_ee.h"
#include "utils/ucc_mpool.h"

#ifndef UCC_TL_SELF_DEFAULT_SCORE
#define UCC_TL_SELF_DEFAULT_SCORE 50
#endif

#ifdef HAVE_PROFILING_TL_SELF
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_SELF_PROFILE_FUNC          UCC_PROFILE_FUNC
#define UCC_TL_SELF_PROFILE_FUNC_VOID     UCC_PROFILE_FUNC_VOID
#define UCC_TL_SELF_PROFILE_REQUEST_NEW   UCC_PROFILE_REQUEST_NEW
#define UCC_TL_SELF_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_SELF_PROFILE_REQUEST_FREE  UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_self_iface {
    ucc_tl_iface_t super;
} ucc_tl_self_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_self_iface_t ucc_tl_self;

typedef struct ucc_tl_self_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_self_lib_config_t;

typedef struct ucc_tl_self_context_config {
    ucc_tl_context_config_t super;
} ucc_tl_self_context_config_t;

typedef struct ucc_tl_self_lib {
    ucc_tl_lib_t             super;
    ucc_tl_self_lib_config_t cfg;
} ucc_tl_self_lib_t;
UCC_CLASS_DECLARE(ucc_tl_self_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_self_context {
    ucc_tl_context_t             super;
    ucc_tl_self_context_config_t cfg;
    ucc_mpool_t                  req_mp;
} ucc_tl_self_context_t;
UCC_CLASS_DECLARE(ucc_tl_self_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_self_task {
    ucc_coll_task_t         super;
    void                   *src;
    void                   *dst;
    size_t                  size;
    ucc_memory_type_t       src_memtype;
    ucc_memory_type_t       dst_memtype;
    ucc_ee_executor_task_t *etask;
} ucc_tl_self_task_t;

typedef struct ucc_tl_self_team {
    ucc_tl_team_t super;
    ucc_status_t  status;
} ucc_tl_self_team_t;
UCC_CLASS_DECLARE(ucc_tl_self_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_SELF_SUPPORTED_COLLS                                            \
    (UCC_COLL_TYPE_ALLTOALL | UCC_COLL_TYPE_ALLTOALLV |                        \
     UCC_COLL_TYPE_ALLGATHER | UCC_COLL_TYPE_ALLGATHERV |                      \
     UCC_COLL_TYPE_ALLREDUCE | UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_BARRIER |   \
     UCC_COLL_TYPE_REDUCE | UCC_COLL_TYPE_FANIN | UCC_COLL_TYPE_FANOUT |       \
     UCC_COLL_TYPE_GATHER | UCC_COLL_TYPE_GATHERV | UCC_COLL_TYPE_SCATTER |    \
     UCC_COLL_TYPE_SCATTERV | UCC_COLL_TYPE_REDUCE_SCATTER |                   \
     UCC_COLL_TYPE_REDUCE_SCATTERV)

#define UCC_TL_SELF_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_self_lib_t))

#define UCC_TL_SELF_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_self_context_t))

ucc_status_t ucc_tl_self_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task_h);
ucc_status_t ucc_tl_self_coll_finalize(ucc_coll_task_t *coll_task);

#endif
