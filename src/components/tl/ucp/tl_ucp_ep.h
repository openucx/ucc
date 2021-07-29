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
#include "core/ucc_team.h"

typedef struct ucc_tl_ucp_context ucc_tl_ucp_context_t;
typedef struct ucc_tl_ucp_team    ucc_tl_ucp_team_t;

ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t         *team,
                                        ucc_rank_t                 team_rank,
                                        ucp_ep_h                  *ep);

ucc_status_t ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx);

static inline ucc_context_addr_header_t *
ucc_tl_ucp_get_team_ep_header(ucc_tl_ucp_team_t *team, ucc_rank_t rank)

{
    return ucc_get_team_ep_header(UCC_TL_CORE_CTX(team), team->super.super.team,
                                  rank);
}

static inline ucc_context_id_t
ucc_tl_ucp_get_rank_key(ucc_tl_ucp_team_t *team, ucc_rank_t rank)
{
    return ucc_tl_ucp_get_team_ep_header(team, rank)->ctx_id;
}

static inline ucc_status_t ucc_tl_ucp_get_ep(ucc_tl_ucp_team_t *team, ucc_rank_t rank,
                                             ucp_ep_h *ep)
{
    ucc_tl_ucp_context_t      *ctx      = UCC_TL_UCP_TEAM_CTX(team);
    ucc_context_addr_header_t *h        = NULL;
    ucc_rank_t                 ctx_rank = 0;
    ucc_status_t               status;

    if (ctx->eps) {
        ctx_rank = ucc_get_ctx_rank(team->super.super.team, rank);
        *ep      = ctx->eps[ctx_rank];
    } else {
        h   = ucc_tl_ucp_get_team_ep_header(team, rank);
        *ep = tl_ucp_hash_get(ctx->ep_hash, h->ctx_id);
    }
    if (NULL == (*ep)) {
        /* Not connected yet */
        status = ucc_tl_ucp_connect_team_ep(team, rank, ep);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to connect team ep");
            *ep = NULL;
            return status;
        }
        if (ctx->eps) {
            ctx->eps[ctx_rank] = *ep;
        } else {
            tl_ucp_hash_put(ctx->ep_hash, h->ctx_id, *ep);
        }
    }
    return UCC_OK;
}

#endif
