#include "ucc_ucp_ctx.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include <ucp/api/ucp.h>

#include <sys/types.h>
#include <unistd.h>

typedef struct ucc_ucp_ctx_handle {
    ucc_ucp_ctx_t      ctx;
    int                ref_count;
} ucc_ucp_ctx_handle_t;

static void ucc_ucp_req_init(void *request)
{
    ucc_ucp_req_t *req = (ucc_ucp_req_t *)request;
    req->status        = UCC_OPERATION_INITIALIZED;
}

static void ucc_ucp_req_cleanup(void *request)
{
}

ucc_status_t ucc_ucp_ctx_create(const ucc_ucp_ctx_create_params_t *params,
                                ucc_ucp_ctx_handle_t **ctx)
{
    ucc_status_t          ucc_status = UCC_OK;
    ucc_ucp_ctx_handle_t *ctx_h;
    ucp_worker_params_t   worker_params;
    ucp_worker_attr_t     worker_attr;
    ucp_params_t          ucp_params;
    ucp_config_t         *ucp_config;
    ucp_context_h         ucp_context;
    ucp_worker_h          ucp_worker;
    ucs_status_t          status;

    UCC_STATIC_ASSERT(UCC_UCP_CTX_LAST - 1 < UCC_BIT(UCC_UCP_CTX_BITS));

    ctx_h = ucc_malloc(sizeof(*ctx_h), "ucc_ucp_ctx");
    if (!ctx_h) {
        ucc_error("failed to allocate %zd bytes for ucc_ucp_ctx",
                  sizeof(*ctx_h));
        return UCC_ERR_NO_MESSAGE;
    }
    status = ucp_config_read(params->prefix, NULL, &ucp_config);
    if (UCS_OK != status) {
        ucc_error("failed to read ucp configuration, %s",
                  ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_cfg_read;
    }

    ucp_params.field_mask =
        UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
        UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP |
        UCP_PARAM_FIELD_TAG_SENDER_MASK;
    ucp_params.features          = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
    ucp_params.request_size      = sizeof(ucc_ucp_req_t);
    ucp_params.request_init      = ucc_ucp_req_init;
    ucp_params.request_cleanup   = ucc_ucp_req_cleanup;
    ucp_params.tag_sender_mask   = UCC_UCP_TAG_SENDER_MASK;

    if (params->mask & UCC_UCP_CTX_PARAM_FIELD_ESTIMATED_NUM_PPN) {
        ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;
        ucp_params.estimated_num_ppn = params->estimated_num_ppn;
    }

    if (params->mask & UCC_UCP_CTX_PARAM_FIELD_ESTIMATED_NUM_EPS) {
        ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        ucp_params.estimated_num_eps = params->estimated_num_eps;
    }

    if (params->mask & UCC_UCP_CTX_PARAM_FIELD_DEVICES) {
        ucp_config_modify(ucp_config, "NET_DEVICES", params->devices);
    }

    status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    ucp_config_release(ucp_config);
    if (UCS_OK != status) {
        ucc_error("failed to init ucp context, %s", ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_cfg_read;
    }

    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    ucc_assert(params->mask & UCC_UCP_CTX_PARAM_FIELD_THREAD_MODE);
    switch (params->thread_mode) {
    case UCC_THREAD_SINGLE:
    case UCC_THREAD_FUNNELED:
        worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
        break;
    case UCC_THREAD_MULTIPLE:
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
        break;
    default:
        /* unreachable */
        ucc_assert(0);
        break;
    }
    status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
    if (UCS_OK != status) {
        ucc_error("failed to create ucp worker, %s", ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_worker_create;
    }

    if (params->thread_mode == UCC_THREAD_MULTIPLE) {
        worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
        ucp_worker_query(ucp_worker, &worker_attr);
        if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
            ucc_error("thread mode multiple is not supported by ucp worker");
            status = UCC_ERR_NOT_SUPPORTED;
            goto err_thread_mode;
        }
    }

    ctx_h->ctx.ucp_context = (void *)ucp_context;
    ctx_h->ctx.ucp_worker  = (void *)ucp_worker;
    ctx_h->ref_count       = 0;
    *ctx                   = ctx_h;
    return UCC_OK;

err_thread_mode:
    ucp_worker_destroy(ucp_worker);
err_worker_create:
    ucp_cleanup(ucp_context);
err_cfg_read:
    free(ctx_h);
    return ucc_status;
}

ucc_status_t ucc_ucp_ctx_destroy(ucc_ucp_ctx_handle_t *ctx_h)
{
    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;
    if (ctx_h->ref_count != 0) {
        ucc_warn(
            "destroying ucc_ucp_ctx %p which is still in use, ref_count %d",
            ctx_h, ctx_h->ref_count);
    }
    ucp_context = (ucp_context_h)ctx_h->ctx.ucp_context;
    ucp_worker  = (ucp_worker_h)ctx_h->ctx.ucp_worker;

    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);
    free(ctx_h);

    return UCC_OK;
}

ucc_status_t ucc_ucp_ctx_get(ucc_ucp_ctx_handle_t *ctx_handle,
                             ucc_ucp_ctx_t **ctx)
{
    ctx_handle->ref_count++;
    *ctx = &ctx_handle->ctx;
    return UCC_OK;
}

ucc_status_t ucc_ucp_ctx_put(ucc_ucp_ctx_handle_t *ctx_handle)
{
    ctx_handle->ref_count--;
    return UCC_OK;
}

ucc_status_t ucc_ucp_ctx_put(ucc_ucp_ctx_handle_t *ctx_handle);

ucc_ucp_ctx_iface_t ucc_ucp_ctx = {
    .super.name = "ucp_ctx",
    .create     = ucc_ucp_ctx_create,
    .destroy    = ucc_ucp_ctx_destroy,
};
