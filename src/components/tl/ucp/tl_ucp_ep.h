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
#include "tl_ucp.h"
#include "utils/ucc_math.h"
typedef struct ucc_tl_ucp_context ucc_tl_ucp_context_t;
typedef struct ucc_tl_ucp_team    ucc_tl_ucp_team_t;
ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t *team, int team_rank);
ucc_status_t ucc_tl_ucp_connect_ctx_ep(ucc_tl_ucp_context_t *ctx, int ctx_rank);

ucc_status_t ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx, ucp_ep_h *eps,
                                  int n_eps);

static inline ucc_status_t ucc_tl_ucp_get_ep(ucc_tl_ucp_team_t *team, int rank,
                                             ucp_ep_h *ep)
{
    ucc_status_t          status;
    ucc_tl_ucp_context_t *ctx;
    if (!team->context_ep_storage) {
        ucc_assert(team->eps);
        if (NULL == team->eps[rank]) {
            /* Not connected yet */
            status = ucc_tl_ucp_connect_team_ep(team, rank);
            if (UCC_OK != status) {
                tl_error(UCC_TL_TEAM_LIB(team), "failed to connect team ep");
                *ep = NULL;
                return status;
            }
        }
        ucc_assert(team->eps[rank]);
        *ep = team->eps[rank];
    } else {
        ctx = UCC_TL_UCP_TEAM_CTX(team);
        ucc_assert(ctx->eps);
        uint32_t ctx_rank = ucc_ep_map_eval(team->ep_map, rank);
        if (NULL == ctx->eps[ctx_rank]) {
            /* Not connected yet */
            status = ucc_tl_ucp_connect_ctx_ep(ctx, ctx_rank);
            if (UCC_OK != status) {
                tl_error(UCC_TL_TEAM_LIB(team), "failed to connect ctx ep");
                *ep = NULL;
                return status;
            }
        }
        ucc_assert(ctx->eps[ctx_rank]);
        *ep = ctx->eps[ctx_rank];
    }
    return UCC_OK;
}

#endif
