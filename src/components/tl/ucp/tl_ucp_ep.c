/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

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
                                                 int is_service, ucp_ep_h *ep,
                                                 void *ucp_address)
{
    ucp_worker_h worker =
        (is_service) ? ctx->service_worker.ucp_worker : ctx->worker.ucp_worker;
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
    status = ucp_ep_create(worker, &ep_params, ep);

    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(ctx->super.super.lib, "ucp returned connect error: %s",
                 ucs_status_string(status));
        return ucs_status_to_ucc_status(status);
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_connect_team_ep(ucc_tl_ucp_team_t *team,
                                        ucc_rank_t core_rank, ucp_ep_h *ep)
{
    ucc_tl_ucp_context_t *ctx = UCC_TL_UCP_TEAM_CTX(team);
    int                   use_service_worker = USE_SERVICE_WORKER(team);
    void                 *addr;

    addr = ucc_get_team_ep_addr(UCC_TL_CORE_CTX(team), UCC_TL_CORE_TEAM(team),
                                core_rank, ucc_tl_ucp.super.super.id);
    addr = use_service_worker ? TL_UCP_EP_ADDR_WORKER_SERVICE(addr)
                              : TL_UCP_EP_ADDR_WORKER(addr);

    return ucc_tl_ucp_connect_ep(ctx, use_service_worker, ep, addr);
}

/* Finds next non-NULL ep in the storage and returns that handle
   for closure. In case of "hash" storage it pops the item,
   in case of "array" sets it to NULL */
static inline ucp_ep_h get_next_ep_to_close(ucc_tl_ucp_worker_t * worker,
                                            ucc_tl_ucp_context_t *ctx, int *i)
{
    ucp_ep_h   ep = NULL;
    ucc_rank_t size;

    if (worker->eps) {
        size = (ucc_rank_t)ctx->super.super.ucc_context->params.oob.n_oob_eps;
        while (NULL == ep && (*i) < size) {
            ep              = worker->eps[*i];
            worker->eps[*i] = NULL;
            (*i)++;
        }
    } else {
        ep = tl_ucp_hash_pop(worker->ep_hash);
    }
    return ep;
}

void ucc_tl_ucp_close_eps(ucc_tl_ucp_worker_t * worker,
                          ucc_tl_ucp_context_t *ctx)
{
     int                          i = 0;
     ucp_ep_h                     ep;
     ucs_status_t                 status;
     ucs_status_ptr_t             close_req;
     ucp_request_param_t          param;

     param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
     param.flags        = 0; // 0 means FLUSH
     ep                 = get_next_ep_to_close(worker, ctx, &i);
     while (ep) {
         close_req = ucp_ep_close_nbx(ep, &param);

         if (UCS_PTR_IS_PTR(close_req)) {
             do {
                 ucp_worker_progress(ctx->worker.ucp_worker);
                 if (ctx->cfg.service_worker != 0) {
                     ucp_worker_progress(ctx->service_worker.ucp_worker);
                 }
                 status = ucp_request_check_status(close_req);
             } while (status == UCS_INPROGRESS);
             ucp_request_free(close_req);
         } else {
             status = UCS_PTR_STATUS(close_req);
         }
         ucc_assert(status <= UCS_OK);
         if (status != UCS_OK) {
             tl_error(ctx->super.super.lib,
                      "error during ucp ep close, ep %p, status %s",
                      ep, ucs_status_string(status));
         }
         ep = get_next_ep_to_close(worker, ctx, &i);
     }
}
