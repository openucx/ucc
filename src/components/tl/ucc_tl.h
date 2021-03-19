/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_TL_H_
#define UCC_TL_H_

#include "components/base/ucc_base_iface.h"

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

typedef enum ucc_tl_type {
    UCC_TL_UCP  = UCC_BIT(0),
    UCC_TL_NCCL = UCC_BIT(1),
} ucc_tl_type_t;

typedef struct ucc_tl_lib_config {
    ucc_base_config_t  super;
    ucc_tl_iface_t    *iface;
} ucc_tl_lib_config_t;
extern ucc_config_field_t ucc_tl_lib_config_table[];

typedef struct ucc_tl_context_config {
    ucc_base_config_t super;
    ucc_tl_lib_t     *tl_lib;
} ucc_tl_context_config_t;
extern ucc_config_field_t ucc_tl_context_config_table[];

ucc_status_t ucc_tl_context_config_read(ucc_tl_lib_t *tl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_tl_context_config_t **cl_config);

ucc_status_t ucc_tl_lib_config_read(ucc_tl_iface_t *iface,
                                    const char *full_prefix,
                                    ucc_tl_lib_config_t **cl_config);

typedef struct ucc_tl_iface {
    ucc_component_iface_t          super;
    ucc_tl_type_t                  type;
    ucs_config_global_list_entry_t tl_lib_config;
    ucs_config_global_list_entry_t tl_context_config;
    ucc_base_lib_iface_t           lib;
    ucc_base_context_iface_t       context;
    ucc_base_team_iface_t          team;
    ucc_base_coll_iface_t          coll;
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
UCC_CLASS_DECLARE(ucc_tl_context_t, ucc_tl_lib_t *, ucc_context_t *);

typedef struct ucc_tl_team {
    ucc_base_team_t super;
} ucc_tl_team_t;
UCC_CLASS_DECLARE(ucc_tl_team_t, ucc_tl_context_t *);

#define UCC_TL_IFACE_DECLARE(_name, _NAME)                                     \
    UCC_BASE_IFACE_DECLARE(TL_, tl_, _name, _NAME)

ucc_status_t ucc_tl_context_get(ucc_context_t *ctx, ucc_tl_type_t type,
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
    } descs[1];
} ucc_team_multiple_req_t;

ucc_status_t
ucc_team_multiple_req_alloc(ucc_team_multiple_req_t **req,
                                   int n_teams);

ucc_status_t ucc_tl_team_create_multiple(ucc_team_multiple_req_t *req);
ucc_status_t ucc_tl_team_destroy_multiple(ucc_team_multiple_req_t *req);

void ucc_team_multiple_req_free(ucc_team_multiple_req_t *req);

#define UCC_TL_CTX_IFACE(_tl_ctx)                                              \
    (ucc_derived_of((_tl_ctx)->super.lib, ucc_tl_lib_t))->iface

#define UCC_TL_TEAM_IFACE(_tl_team)                                            \
    (ucc_derived_of((_tl_team)->super.context->lib, ucc_tl_lib_t))->iface

#define UCC_TL_TEAM_LIB(_tl_team) (_tl_team)->super.super.context->lib

#define UCC_TL_CORE_CTX(_tl_team) ((_tl_team)->super.super.context->ucc_context)

#endif
