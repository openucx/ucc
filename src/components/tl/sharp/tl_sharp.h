/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_SHARP_H_
#define UCC_TL_SHARP_H_

#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"
#include "utils/ucc_rcache.h"

#include <sharp/api/sharp.h>
#include <limits.h>

#ifndef UCC_TL_SHARP_DEFAULT_SCORE
#define UCC_TL_SHARP_DEFAULT_SCORE 30
#endif

#ifdef HAVE_PROFILING_TL_SHARP
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_SHARP_PROFILE_FUNC UCC_PROFILE_FUNC
#define UCC_TL_SHARP_PROFILE_FUNC_VOID UCC_PROFILE_FUNC_VOID
#define UCC_TL_SHARP_PROFILE_REQUEST_NEW UCC_PROFILE_REQUEST_NEW
#define UCC_TL_SHARP_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_SHARP_PROFILE_REQUEST_FREE UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_sharp_iface {
    ucc_tl_iface_t super;
} ucc_tl_sharp_iface_t;

extern ucc_tl_sharp_iface_t ucc_tl_sharp;

typedef struct ucc_tl_sharp_lib_config {
    ucc_tl_lib_config_t super;
    int                 use_internal_oob;
} ucc_tl_sharp_lib_config_t;

typedef struct ucc_tl_sharp_context_config {
    ucc_tl_context_config_t  super;
    struct sharp_coll_config cfg;
    char                    *dev_list;
    int                      use_rcache;
    size_t                   reg_threshold;
    unsigned int             rand_seed;
    unsigned int             uprogress_num_polls;
    int                      context_per_team;
    int                      enable_lazy_group_alloc;
    int                      team_max_ppn;
    int                      use_multi_channel;
} ucc_tl_sharp_context_config_t;

typedef struct ucc_tl_sharp_lib {
    ucc_tl_lib_t              super;
    ucc_tl_sharp_lib_config_t cfg;
} ucc_tl_sharp_lib_t;
UCC_CLASS_DECLARE(ucc_tl_sharp_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_sharp_oob_ctx {
    void           *ctx;
    union {
        ucc_oob_coll_t *oob;
        ucc_subset_t    subset;
    };
} ucc_tl_sharp_oob_ctx_t;

typedef struct ucc_tl_sharp_reg {
    void *mr;
} ucc_tl_sharp_reg_t;

typedef struct ucc_tl_sharp_rcache_region {
    ucc_rcache_region_t super;
    ucc_tl_sharp_reg_t  reg;
} ucc_tl_sharp_rcache_region_t;

typedef struct ucc_tl_sharp_context {
    ucc_tl_context_t              super;
    ucc_thread_mode_t             tm;
    struct sharp_coll_context    *sharp_context;
    ucc_tl_sharp_context_config_t cfg;
    ucc_mpool_t                   req_mp;
    ucc_tl_sharp_oob_ctx_t        oob_ctx;
    ucc_rcache_t                 *rcache;
    struct sharp_coll_caps        sharp_caps;
} ucc_tl_sharp_context_t;
UCC_CLASS_DECLARE(ucc_tl_sharp_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_sharp_team {
    ucc_tl_team_t             super;
    struct sharp_coll_context *sharp_context;
    ucc_rcache_t              *rcache;
    struct sharp_coll_comm    *sharp_comm;
    ucc_tl_sharp_oob_ctx_t    oob_ctx;
    ucc_topo_t                *topo;
} ucc_tl_sharp_team_t;

typedef struct ucc_tl_sharp_task {
    ucc_coll_task_t             super;
    void                       *req_handle;
    union {
        struct {
            ucc_tl_sharp_reg_t *s_mem_h;
            ucc_tl_sharp_reg_t *r_mem_h;
        } allgather;
        struct {
            ucc_tl_sharp_reg_t *s_mem_h;
            ucc_tl_sharp_reg_t *r_mem_h;
        } allreduce;
        struct {
            ucc_tl_sharp_reg_t *s_mem_h;
            ucc_tl_sharp_reg_t *r_mem_h;
        } reduce_scatter;
        struct {
            ucc_tl_sharp_reg_t *mem_h;
        } bcast;
    };
} ucc_tl_sharp_task_t;

ucc_status_t ucc_tl_sharp_context_init(ucc_tl_sharp_context_t *sharp_ctx,
                                       struct sharp_coll_context **context,
                                       ucc_tl_sharp_oob_ctx_t *oob_ctx,
                                       ucc_topo_t *topo);
ucc_status_t ucc_tl_sharp_rcache_create(struct sharp_coll_context *contex,
                                        ucc_rcache_t **rcache);

ucc_status_t sharp_status_to_ucc_status(int status);

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_sharp_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_sharp_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_sharp_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

#define UCC_TL_BASIC_SHARP_SUPPORTED_COLLS                                     \
    (UCC_COLL_TYPE_ALLREDUCE | UCC_COLL_TYPE_BARRIER | UCC_COLL_TYPE_BCAST)


#define UCC_TL_SHARP_SUPPORTED_COLLS \
    (UCC_TL_BASIC_SHARP_SUPPORTED_COLLS | \
    (HAVE_DECL_SHARP_COLL_DO_REDUCE_SCATTER ? UCC_COLL_TYPE_REDUCE_SCATTER : 0) | \
    (HAVE_DECL_SHARP_COLL_DO_ALLGATHER ? UCC_COLL_TYPE_ALLGATHER : 0))

UCC_CLASS_DECLARE(ucc_tl_sharp_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_SHARP_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_sharp_lib_t))

#endif
