/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_TL_H_
#define UCC_TL_H_

#include "components/base/ucc_base_iface.h"
#include "core/ucc_context.h"
#include "utils/ucc_sys.h"

/** TL (transport layer) is an internal interface that provides a basic
    implementation of collectives and p2p primitives. It differs from CL in that
    it focuses on the abstraction over hardware transport rather than the
    programming model. These collectives can leverage either p2p communication
    transport or collective transport. The primary example of p2p communication
    transport is UCX, and the primary example of collective transport is SHARP,
    MCAST, and shared memory.
 */

typedef struct ucc_tl_lib     ucc_tl_lib_t;
typedef struct ucc_tl_iface   ucc_tl_iface_t;
typedef struct ucc_tl_context ucc_tl_context_t;
typedef struct ucc_tl_team    ucc_tl_team_t;

typedef struct ucc_tl_lib_config {
    ucc_base_lib_config_t  super;
    ucc_tl_iface_t        *iface;
} ucc_tl_lib_config_t;
extern ucc_config_field_t ucc_tl_lib_config_table[];

typedef struct ucc_tl_context_config {
    ucc_base_ctx_config_t super;
    ucc_tl_lib_t         *tl_lib;
} ucc_tl_context_config_t;
extern ucc_config_field_t ucc_tl_context_config_table[];

ucc_status_t ucc_tl_context_config_read(ucc_tl_lib_t *tl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_tl_context_config_t **cl_config);

ucc_status_t ucc_tl_lib_config_read(ucc_tl_iface_t *iface,
                                    const char *full_prefix,
                                    ucc_tl_lib_config_t **cl_config);

typedef struct ucc_tl_service_coll {
    ucc_status_t (*allreduce)(ucc_base_team_t *team, void *sbuf, void *rbuf,
                              ucc_datatype_t dt, size_t count,
                              ucc_reduction_op_t op, ucc_subset_t subset,
                              ucc_coll_task_t **task);
    ucc_status_t (*allgather)(ucc_base_team_t *team, void *sbuf, void *rbuf,
                              size_t msgsize, ucc_subset_t subset,
                              ucc_coll_task_t **task);
    ucc_status_t (*bcast)(ucc_base_team_t *team, void *buf, size_t msgsize,
                          ucc_rank_t root, ucc_subset_t subset,
                          ucc_coll_task_t **task);
    void         (*update_id)(ucc_base_team_t *team, uint16_t id);
} ucc_tl_service_coll_t;

typedef struct ucc_tl_coll_plugin_iface {
    ucc_component_iface_t          super;
    ucs_config_global_list_entry_t config;
    ucc_get_coll_scores_fn_t       get_scores;
    uint32_t                       id;
} ucc_tl_coll_plugin_iface_t;

typedef struct ucc_tl_iface {
    ucc_component_iface_t          super;
    ucs_config_global_list_entry_t tl_lib_config;
    ucs_config_global_list_entry_t tl_context_config;
    ucc_base_lib_iface_t           lib;
    ucc_base_context_iface_t       context;
    ucc_base_team_iface_t          team;
    ucc_base_coll_iface_t          coll;
    ucc_tl_service_coll_t          scoll;
    ucc_base_coll_alg_info_t *     alg_info[UCC_COLL_TYPE_NUM];
    ucc_component_framework_t      coll_plugins;
} ucc_tl_iface_t;

typedef struct ucc_tl_lib {
    ucc_base_lib_t              super;
    ucc_tl_iface_t             *iface;
} ucc_tl_lib_t;
UCC_CLASS_DECLARE(ucc_tl_lib_t, ucc_tl_iface_t *, const ucc_tl_lib_config_t *);

typedef struct ucc_tl_context {
    ucc_base_context_t super;
    int                ref_count;
} ucc_tl_context_t;
UCC_CLASS_DECLARE(ucc_tl_context_t, const ucc_tl_context_config_t *,
                  ucc_context_t *);

typedef struct ucc_tl_team {
    ucc_base_team_t super;
} ucc_tl_team_t;
UCC_CLASS_DECLARE(ucc_tl_team_t, ucc_tl_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_IFACE_DECLARE(_name, _NAME)                                     \
    UCC_BASE_IFACE_DECLARE(TL_, tl_, _name, _NAME)

ucc_status_t ucc_tl_context_get(ucc_context_t *ctx, const char *name,
                                ucc_tl_context_t **tl_context);
ucc_status_t ucc_tl_context_put(ucc_tl_context_t *tl_context);

typedef struct ucc_team_multiple_req {
    int                          n_teams;
    int                          last;
    struct ucc_team_team_desc {
        ucc_tl_context_t        *ctx;
        ucc_tl_team_t           *team;
        ucc_base_team_params_t   param;
        ucc_status_t             status;
        uint64_t                 args[2];
    } descs[1];
} ucc_team_multiple_req_t;

ucc_status_t
ucc_team_multiple_req_alloc(ucc_team_multiple_req_t **req,
                                   int n_teams);

ucc_status_t ucc_tl_team_create_multiple(ucc_team_multiple_req_t *req);
ucc_status_t ucc_tl_team_destroy_multiple(ucc_team_multiple_req_t *req);

void ucc_team_multiple_req_free(ucc_team_multiple_req_t *req);

typedef struct ucc_tl_lib_attr {
    ucc_base_lib_attr_t super;
} ucc_tl_lib_attr_t;

#define UCC_TL_CTX_IFACE(_tl_ctx)                                              \
    (ucc_derived_of((_tl_ctx)->super.lib, ucc_tl_lib_t))->iface

#define UCC_TL_TEAM_IFACE(_tl_team)                                            \
    (ucc_derived_of((_tl_team)->super.context->lib, ucc_tl_lib_t))->iface

#define UCC_TL_TEAM_LIB(_tl_team) (_tl_team)->super.super.context->lib

#define UCC_TL_TEAM_CTX(_tl_team) (_tl_team)->super.super.context

#define UCC_TL_CORE_CTX(_tl_team) ((_tl_team)->super.super.context->ucc_context)

#define UCC_TL_CTX_OOB(_ctx) ((_ctx)->super.super.ucc_context->params.oob)

#define UCC_TL_TEAM_SIZE(_tl_team) (_tl_team)->super.super.params.size

#define UCC_TL_TEAM_RANK(_tl_team) (_tl_team)->super.super.params.rank

#define UCC_TL_CORE_TEAM(_tl_team) (_tl_team)->super.super.params.team

#define UCC_TL_TEAM_MAP(_tl_team) (_tl_team)->super.super.params.map

#define UCC_TL_TEAM_OOB(_tl_team) (_tl_team)->super.super.params.params.oob
#endif
