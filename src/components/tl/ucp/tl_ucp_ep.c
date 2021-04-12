#include "tl_ucp.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_addr.h"

//NOLINTNEXTLINE
static void ucc_tl_ucp_err_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    /* Dummy fn - we don't expect errors in disconnect flow */
    ;
}

static inline ucc_status_t ucc_tl_ucp_connect_ep(ucc_tl_ucp_context_t *ctx,
                                                 ucp_ep_h *ep, char *addr_array,
                                                 size_t max_addrlen, ucc_rank_t rank)
{
    ucc_tl_ucp_addr_t  *address = (ucc_tl_ucp_addr_t *)(addr_array +
                                                        max_addrlen * rank);
    ucp_ep_params_t ep_params;
    ucs_status_t    status;
    if (*ep) {
        /* Already connected */
        return UCC_OK;
    }
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = (ucp_address_t*)address->addr;

    if (!UCC_TL_CTX_HAS_OOB(ctx)) {
        ep_params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        ep_params.err_handler.cb  = ucc_tl_ucp_err_handler;
        ep_params.err_handler.arg = NULL;
        ep_params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLER;
    }
    status = ucp_ep_create(ctx->ucp_worker, &ep_params, ep);

    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(ctx->super.super.lib, "ucp returned connect error: %s",
                 ucs_status_string(status));
        return ucs_status_to_ucc_status(status);
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t *team, ucc_rank_t team_rank,
                                        ucc_context_id_t key, ucp_ep_h *ep)
{
    ucc_tl_ucp_context_t *ctx = UCC_TL_UCP_TEAM_CTX(team);
    ucc_status_t          status;
    status = ucc_tl_ucp_connect_ep(ctx, ep,
                                   (char*)team->addr_storage->addresses,
                                   team->addr_storage->max_addrlen, team_rank);
    if (UCC_OK == status) {
        tl_ucp_hash_put(ctx->ep_hash, key, *ep);
    }
    return status;
}

ucc_status_t ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx)
{
    ucc_tl_ucp_ep_close_state_t *state = &ctx->ep_close_state;
    ucp_ep_h                     ep;
    ucs_status_t                 status;
    if (state->close_req) {
        ucp_worker_progress(ctx->ucp_worker);
        status = ucp_request_check_status(state->close_req);
        if (status != UCS_OK) {
            return UCC_INPROGRESS;
        }
        ucp_request_free(state->close_req);
    }
    ep = tl_ucp_hash_pop(ctx->ep_hash);
    while (ep) {
        state->close_req =
            ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
        if (ucc_unlikely(UCS_PTR_IS_ERR(state->close_req))) {
            tl_error(ctx->super.super.lib, "failed to start ep close, ep %p",
                     ep);
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
        ep = tl_ucp_hash_pop(ctx->ep_hash);
    }
    state->close_req = NULL;
    return UCC_OK;
}
