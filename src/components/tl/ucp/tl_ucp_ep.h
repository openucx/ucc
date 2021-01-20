/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"

#ifndef UCC_TL_UCP_EP_H_
#define UCC_TL_UCP_EP_H_
#include "ucc/api/ucc.h"
#include <ucp/api/ucp.h>

typedef struct ucc_tl_ucp_context ucc_tl_ucp_context_t;
typedef struct ucc_tl_ucp_team    ucc_tl_ucp_team_t;
ucc_status_t ucc_tl_ucp_connect_ep(ucc_tl_ucp_context_t *ctx,
                                   ucc_tl_ucp_team_t *team, char *addr_array,
                                   size_t max_addrlen, int rank);

typedef struct ucc_tl_ucp_ep_close_state {
    int   ep;
    void *close_req;
} ucc_tl_ucp_ep_close_state_t;

ucc_status_t ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx, ucp_ep_h *eps,
                                  int n_eps);
#endif
