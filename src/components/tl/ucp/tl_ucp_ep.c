#include "tl_ucp.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_addr.h"

static inline ucc_status_t ucc_tl_ucp_connect_ep(ucc_tl_ucp_context_t *ctx,
                                                 ucp_ep_h *ep, char *addr_array,
                                                 size_t max_addrlen, int rank)
{
    ucp_address_t  *address = (ucp_address_t *)(addr_array + max_addrlen * rank);
    ucp_ep_params_t ep_params;
    ucs_status_t    status;
    if (*ep) {
        /* Already connected */
        return UCC_OK;
    }
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = address;

    status = ucp_ep_create(ctx->ucp_worker, &ep_params, ep);

    if (UCS_OK != status) {
        tl_error(ctx->super.super.lib, "ucp returned connect error: %s",
                 ucs_status_string(status));
        return ucs_status_to_ucc_status(status);
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_connect_ctx_ep(ucc_tl_ucp_context_t *ctx, int ctx_rank)
{
    return ucc_tl_ucp_connect_ep(ctx, &ctx->eps[ctx_rank],
                                 ctx->addr_storage->addresses,
                                 ctx->addr_storage->max_addrlen, ctx_rank);
}

ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t *team, int team_rank)
{
    ucc_tl_ucp_context_t *ctx = UCC_TL_UCP_TEAM_CTX(team);
    if (team->context_ep_storage) {
        ucc_assert(team->ep_map.type > 0);
        uint32_t ctx_rank = ucc_ep_map_eval(team->ep_map, team_rank);
        return ucc_tl_ucp_connect_ctx_ep(ctx, ctx_rank);
    }
    return ucc_tl_ucp_connect_ep(ctx, &team->eps[team_rank],
                                 team->addr_storage->addresses,
                                 team->addr_storage->max_addrlen, team_rank);
}

ucc_status_t ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx, ucp_ep_h *eps,
                                  int n_eps)
{
    ucc_tl_ucp_ep_close_state_t *state = &ctx->ep_close_state;
    ucs_status_t status;
    if (state->close_req) {
        ucp_worker_progress(ctx->ucp_worker);
        status = ucp_request_check_status(state->close_req);
        if (status != UCS_OK) {
            return UCC_INPROGRESS;
        }
        ucp_request_free(state->close_req);
        eps[state->ep++] = NULL;
    }

    for (; state->ep < n_eps; state->ep++) {
        if (!eps[state->ep]) {
            continue;
        }
        state->close_req =
            ucp_ep_close_nb(eps[state->ep], UCP_EP_CLOSE_MODE_FLUSH);
        if (UCS_PTR_IS_ERR(state->close_req)) {
            tl_error(ctx->super.super.lib, "failed to start ep close, ep %p",
                     eps[state->ep]);
        }
        status = UCS_PTR_STATUS(state->close_req);
        /* try progress once */
        if (status != UCS_OK) {
            ucp_worker_progress(ctx->ucp_worker);
            status = ucp_request_check_status(state->close_req);
            if (status != UCS_OK) {
                return UCC_INPROGRESS;
            }
            ucp_request_free(state->close_req);
        }
        if (status == UCS_OK) {
            eps[state->ep] = NULL;
        }
    }
    state->close_req = NULL;
    state->ep        = 0;
    return UCC_OK;
}
