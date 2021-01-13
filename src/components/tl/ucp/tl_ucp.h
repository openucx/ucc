/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_H_
#define UCC_TL_UCP_H_
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
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
    int                     test_param;
} ucc_tl_ucp_context_config_t;

typedef struct ucc_tl_ucp_lib {
    ucc_tl_lib_t super;
} ucc_tl_ucp_lib_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_context {
    ucc_tl_context_t super;
    ucp_context_h    ucp_context;
    ucp_worker_h     ucp_worker;
    size_t           ucp_addrlen;
} ucc_tl_ucp_context_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_ucp_team {
    ucc_tl_team_t super;
} ucc_tl_ucp_team_t;
UCC_CLASS_DECLARE(ucc_tl_ucp_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

typedef struct ucc_tl_ucp_req {
    ucc_status_t status;
} ucc_tl_ucp_req_t;
#endif
