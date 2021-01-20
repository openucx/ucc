#include "tl_ucp.h"
#include "tl_ucp_ep.h"

ucc_status_t ucc_tl_ucp_connect_ep(ucc_tl_ucp_context_t *ctx,
                                   ucc_tl_ucp_team_t *team, char *addr_array,
                                   size_t max_addrlen, int rank)
{
    ucp_address_t  *address = (ucp_address_t *)(addr_array + max_addrlen * rank);
    ucp_ep_params_t ep_params;
    ucs_status_t    status;
    ucp_ep_h       *ep;
    if (team->context_ep_storage) {
        //TODO get ep from ctx storage ctx->eps[rank_to_ctx_rank] using mappers
        ep = NULL;
    } else {
        ucc_assert(team && team->eps);
        ep = &team->eps[rank];
    }
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

ucc_status_t ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx, ucp_ep_h *eps,
                                  int n_eps)
{
    void        *close_req;
    ucs_status_t status;
    int          i;
    for (i = 0; i < n_eps; i++) {
        if (!eps[i]) {
            continue;
        }
        close_req = ucp_ep_close_nb(eps[i], UCP_EP_CLOSE_MODE_FLUSH);
        if (UCS_PTR_IS_ERR(close_req)) {
            tl_error(ctx->super.super.lib, "failed to start ep close, ep %p",
                     eps[i]);
        }
        status = UCS_PTR_STATUS(close_req);
        if (status != UCS_OK) {
            while (status != UCS_OK) {
                ucp_worker_progress(ctx->ucp_worker);
                status = ucp_request_check_status(close_req);
            }
            ucp_request_free(close_req);
        }
    }
    return UCC_OK;
}
