/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* TL/UCP endpoint address layout: (ucp_addrlen may very per proc)

   [ucp_addrlen][ucp_worker_address][onesided_info]
       8 bytes    ucp_addrlen bytes
*/
#define TL_UCP_EP_ADDRLEN_SIZE 8
#define TL_UCP_EP_ADDR_WORKER_LEN(_addr) (*((uint64_t*)(_addr)))
#define TL_UCP_EP_ADDR_WORKER(_addr) PTR_OFFSET((_addr), 8)
#define TL_UCP_EP_ADDR_ONESIDED_INFO(_addr) \
    PTR_OFFSET((_addr), 8 + TL_UCP_EP_ADDR_WORKER_LEN(_addr))

typedef struct ucc_tl_ucp_context ucc_tl_ucp_context_t;
typedef struct ucc_tl_ucp_team    ucc_tl_ucp_team_t;

ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t         *team,
                                        ucc_rank_t                 team_rank,
                                        ucp_ep_h                  *ep);

void ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx);

static inline ucc_context_addr_header_t *
ucc_tl_ucp_get_team_ep_header(ucc_tl_ucp_team_t *team, ucc_rank_t core_rank)

{
    return ucc_get_team_ep_header(UCC_TL_CORE_CTX(team), UCC_TL_CORE_TEAM(team),
                                  core_rank);
}

static inline ucc_status_t ucc_tl_ucp_get_ep(ucc_tl_ucp_team_t *team, ucc_rank_t rank,
                                             ucp_ep_h *ep)
{
    ucc_tl_ucp_context_t      *ctx      = UCC_TL_UCP_TEAM_CTX(team);
    ucc_context_addr_header_t *h        = NULL;
    ucc_rank_t                 ctx_rank = 0;
    ucc_status_t               status;
    ucc_rank_t                 core_rank;

    core_rank = ucc_ep_map_eval(UCC_TL_TEAM_MAP(team), rank);
    if (ctx->eps) {
        ucc_team_t *core_team = UCC_TL_CORE_TEAM(team);
        /* Core super.super.team ptr is NULL for service_team
           which has scope == UCC_CL_LAST + 1*/
        ucc_assert((NULL != core_team) || IS_SERVICE_TEAM(team));
        ctx_rank = core_team ? ucc_get_ctx_rank(core_team, core_rank)
                       : core_rank;
        *ep      = ctx->eps[ctx_rank];
    } else {
        h   = ucc_tl_ucp_get_team_ep_header(team, core_rank);
        *ep = tl_ucp_hash_get(ctx->ep_hash, h->ctx_id);
    }
    if (NULL == (*ep)) {
        /* Not connected yet */
        status = ucc_tl_ucp_connect_team_ep(team, core_rank, ep);
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
