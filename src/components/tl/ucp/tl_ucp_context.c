/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_tag.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_ep.h"
#include <limits.h>

UCC_CLASS_INIT_FUNC(ucc_tl_ucp_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_ucp_context_config_t *tl_ucp_config =
        ucc_derived_of(config, ucc_tl_ucp_context_config_t);
    ucc_status_t        ucc_status = UCC_OK;
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t   worker_attr;
    ucp_params_t        ucp_params;
    ucp_config_t       *ucp_config;
    ucp_context_h       ucp_context;
    ucp_worker_h        ucp_worker;
    ucs_status_t        status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, tl_ucp_config->super.tl_lib,
                              params->context);
    memcpy(&self->cfg, tl_ucp_config, sizeof(*tl_ucp_config));
    self->ep_close_state.close_req = NULL;
    status = ucp_config_read(params->prefix, NULL, &ucp_config);
    if (UCS_OK != status) {
        tl_error(self->super.super.lib, "failed to read ucp configuration, %s",
                 ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_cfg;
    }

    ucp_params.field_mask =
        UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_TAG_SENDER_MASK;
    ucp_params.features        = UCP_FEATURE_TAG;
    ucp_params.tag_sender_mask = UCC_TL_UCP_TAG_SENDER_MASK;

    if (params->estimated_num_ppn > 0) {
        ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;
        ucp_params.estimated_num_ppn = params->estimated_num_ppn;
    }

    if (params->estimated_num_eps > 0) {
        ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        ucp_params.estimated_num_eps = params->estimated_num_eps;
    }

    status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    ucp_config_release(ucp_config);
    if (UCS_OK != status) {
        tl_error(self->super.super.lib, "failed to init ucp context, %s",
                 ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_cfg;
    }

    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
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
        tl_error(self->super.super.lib, "failed to create ucp worker, %s",
                 ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_worker_create;
    }

    if (params->thread_mode == UCC_THREAD_MULTIPLE) {
        worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
        ucp_worker_query(ucp_worker, &worker_attr);
        if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
            tl_error(self->super.super.lib,
                     "thread mode multiple is not supported by ucp worker");
            ucc_status = UCC_ERR_NOT_SUPPORTED;
            goto err_thread_mode;
        }
    }

    self->ucp_context = ucp_context;
    self->ucp_worker  = ucp_worker;
    self->worker_address = NULL;

    ucc_status = ucc_mpool_init(&self->req_mp, sizeof(ucc_tl_ucp_task_t),
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX, NULL, NULL,
                                "tl_ucp_req_mp");
    if (UCC_OK != ucc_status) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_ucp_req mpool");
        goto err_thread_mode;
    }
    if (UCC_OK != ucc_context_progress_register(
                      params->context,
                      (ucc_context_progress_fn_t)ucp_worker_progress,
                      self->ucp_worker)) {
        tl_error(self->super.super.lib, "failed to register progress function");
        goto err_thread_mode;
    }
    self->ep_hash = kh_init(tl_ucp_ep_hash);
    tl_info(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;

err_thread_mode:
    ucp_worker_destroy(ucp_worker);
err_worker_create:
    ucp_cleanup(ucp_context);
err_cfg:
    return ucc_status;
}

static void ucc_tl_ucp_context_barrier(ucc_tl_ucp_context_t *ctx,
                                       ucc_context_oob_coll_t *oob)
{
    char        *rbuf = ucc_malloc(sizeof(char*) * oob->participants,
                                   "tmp_barrier");
    ucc_status_t status;
    char         sbuf;
    void        *req;

    if (!rbuf) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tmp barrier array",
                 sizeof(char*) * oob->participants);
        return;
    }
    if (UCC_OK == oob->allgather(&sbuf, rbuf, sizeof(char), oob->coll_info,
                                 &req)) {
        ucc_assert(req);
        while (UCC_OK != (status = oob->req_test(req))) {
            if (status < 0) {
                tl_error(ctx->super.super.lib, "failed to test oob req");
                break;
            }
        }
        oob->req_free(req);
    }
    ucc_free(rbuf);
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_context_t)
{
    ucc_status_t status;
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    while (UCC_OK != (status = ucc_tl_ucp_close_eps(self))) {
        //TODO can we hang the runtime this way ?
        if (status < 0) {
            tl_error(self->super.super.lib,
                     "failed to close ucp endpoint: %s",
                     ucc_status_string(status));
            break;
        }
    }
    kh_destroy(tl_ucp_ep_hash, self->ep_hash);
    if (UCC_TL_CTX_HAS_OOB(self)) {
        ucc_tl_ucp_context_barrier(self, &UCC_TL_CTX_OOB(self));
    }
    ucc_context_progress_deregister(
        self->super.super.ucc_context,
        (ucc_context_progress_fn_t)ucp_worker_progress, self->ucp_worker);
    ucp_worker_destroy(self->ucp_worker);
    ucc_mpool_cleanup(&self->req_mp, 1);
    ucp_cleanup(self->ucp_context);
}

UCC_CLASS_DEFINE(ucc_tl_ucp_context_t, ucc_tl_context_t);

ucc_status_t ucc_tl_ucp_populate_rcache(void *addr, size_t length,
                                        ucs_memory_type_t mem_type,
                                        ucc_tl_ucp_context_t *ctx)
{
    ucp_mem_map_params_t mmap_params;
    ucp_mem_h            mh;
    ucs_status_t         status;

    mmap_params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                              UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                              UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.address     = addr;
    mmap_params.length      = length;
    mmap_params.memory_type = mem_type;

    /* do map and umap to populate the cache */
    status = ucp_mem_map(ctx->ucp_context, &mmap_params, &mh);
    if (status != UCS_OK) {
        return ucs_status_to_ucc_status(status);
    }

    status = ucp_mem_unmap(ctx->ucp_context, mh);
    if (status != UCS_OK) {
        return ucs_status_to_ucc_status(status);
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                                         ucc_base_attr_t *attr) /* NOLINT */
{
    /* TODO */
    return UCC_ERR_NOT_IMPLEMENTED;
}
