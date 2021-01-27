/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_H_
#define UCC_TL_UCP_H_
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_mpool.h"

#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>

typedef struct ucc_tl_ucp_iface {
    ucc_tl_iface_t super;
} ucc_tl_ucp_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_ucp_iface_t ucc_tl_ucp;

typedef struct ucc_tl_ucp_lib_config {
    ucc_tl_lib_config_t super;
} ucc_tl_ucp_lib_config_t;

typedef struct ucc_tl_ucp_context_config {
    ucc_tl_context_config_t super;
    uint32_t                preconnect;
} ucc_tl_ucp_context_config_t;

typedef struct ucc_tl_ucp_lib {
    ucc_tl_lib_t super;
} ucc_tl_ucp_lib_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_addr_storage ucc_tl_ucp_addr_storage_t;

typedef struct ucc_tl_ucp_ep_close_state {
    int   ep;
    void *close_req;
} ucc_tl_ucp_ep_close_state_t;

typedef struct ucc_tl_ucp_context {
    ucc_tl_context_t            super;
    ucp_context_h               ucp_context;
    ucp_worker_h                ucp_worker;
    size_t                      ucp_addrlen;
    ucp_address_t              *worker_address;
    uint32_t                    preconnect;
    ucc_tl_ucp_ep_close_state_t ep_close_state;
    ucc_mpool_t                 req_mp;
    ucp_ep_h                   *eps;
    ucc_tl_ucp_addr_storage_t  *addr_storage;
} ucc_tl_ucp_context_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_team {
    ucc_tl_team_t              super;
    ucc_status_t               status;
    int                        context_ep_storage; /*< The flag
              indicates whether ucp endpoints are stored on the
              ucc_tl_ucp_context or are they created per-team.
              This optimization is only possible when user provides
              the necessary rank mappings team_rank->context_rank. */
    ucp_ep_h                  *eps;
    int                        size;
    int                        rank;
    ucc_tl_ucp_addr_storage_t *addr_storage;
    uint32_t                   id;
    uint32_t                   scope;
    uint32_t                   scope_id;
} ucc_tl_ucp_team_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_UCP_TEAM_CTX(_team)                                             \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_ucp_context_t))

#define UCC_TL_UCP_TEAM_CORE_CTX(_team)                                        \
    ((_team)->super.super.context->ucc_context)

#endif
