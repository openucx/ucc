#include "tl_ucp.h"
#include "tl_ucp_ep.h"

//NOLINTNEXTLINE
static void ucc_tl_ucp_err_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    /* In case we don't have OOB barrier, errors are expected.
     * This cb will suppress UCX from raising errors*/
    ;
}

static inline ucc_status_t ucc_tl_ucp_connect_ep(ucc_tl_ucp_context_t *ctx,
                                                 ucp_ep_h             *ep,
                                                 void *ucp_address)
{
    ucp_ep_params_t ep_params;
    ucs_status_t    status;
    if (*ep) {
        /* Already connected */
        return UCC_OK;
    }
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = (ucp_address_t *)ucp_address;

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

ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t         *team,
                                        ucc_rank_t                 team_rank,
                                        ucp_ep_h                  *ep)
{
    ucc_tl_ucp_context_t *ctx = UCC_TL_UCP_TEAM_CTX(team);
    void                 *addr;

    addr = ucc_get_team_ep_addr(UCC_TL_CORE_CTX(team), team->super.super.team,
                                team_rank, ucc_tl_ucp.super.super.id);
    return ucc_tl_ucp_connect_ep(ctx, ep, addr);
}

void ucc_tl_ucp_close_eps(ucc_tl_ucp_context_t *ctx)
{
     ucp_ep_h                     ep;
     ucs_status_t                 status;
     void **                      close_reqs;
     void *                       close_req;
     int                          i, close_reqs_counter = 0;

     close_reqs = (void **)ucc_malloc(sizeof(void *) * kh_size(ctx->ep_hash),
                                      "ep close requests array");
     if (!close_reqs) {
         tl_error(ctx->super.super.lib, "Unable to allocate memory");
         return;
     }
     ep = tl_ucp_hash_pop(ctx->ep_hash);
     while (ep) {
         close_req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
         if (UCS_PTR_IS_PTR(close_req)) {
             close_reqs[close_reqs_counter] = close_req;
             close_reqs_counter++;
         }
         else if ((UCS_PTR_STATUS(close_req) != UCS_OK) &&
                  UCC_TL_CTX_HAS_OOB(ctx)) {
             tl_error(ctx->super.super.lib, "failed to close properly, ep %p, error - %s",
                      ep, ucc_status_string(ucs_status_to_ucc_status(UCS_PTR_STATUS(close_req))));
             // In case we have no OOB, we have no barrier to sync closure, and we can ignore errors, which are expected
         }
         ep = tl_ucp_hash_pop(ctx->ep_hash);
     }
     for (i = 0; i < close_reqs_counter; i++) {
         close_req = close_reqs[i];
         // TODO: Should we put a timer? in UCX, some examples have timer, and some don't
         do {
             ucp_worker_progress(ctx->ucp_worker);
             status = ucp_request_check_status(close_req);
         } while (status == UCS_INPROGRESS && close_req);
         if (close_req) {
             close_reqs[i] = NULL;
             ucp_request_free(close_req);
         }
     }
     close_req = NULL;
     ucc_free(close_reqs);
}
