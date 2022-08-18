/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_tag.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_ep.h"
#include "utils/ucc_math.h"
#include "utils/arch/cpu.h"
#include "schedule/ucc_schedule_pipelined.h"
#include <limits.h>

UCC_CLASS_INIT_FUNC(ucc_tl_ucp_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_ucp_context_config_t *tl_ucp_config =
        ucc_derived_of(config, ucc_tl_ucp_context_config_t);
    ucc_status_t        ucc_status = UCC_OK;
    ucp_context_attr_t  context_attr;
    ucp_worker_params_t worker_params;
    ucp_worker_attr_t   worker_attr;
    ucp_params_t        ucp_params;
    ucp_config_t       *ucp_config;
    ucp_context_h       ucp_context;
    ucp_worker_h        ucp_worker;
    ucs_status_t        status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_ucp_config->super,
                              params->context);
    memcpy(&self->cfg, tl_ucp_config, sizeof(*tl_ucp_config));
    status = ucp_config_read(params->prefix, NULL, &ucp_config);
    if (UCS_OK != status) {
        tl_error(self->super.super.lib, "failed to read ucp configuration, %s",
                 ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_cfg;
    }

    ucp_params.field_mask =
        UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_TAG_SENDER_MASK;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_AM;
    if (params->params.mask & UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS) {
        ucp_params.features |= UCP_FEATURE_RMA | UCP_FEATURE_AMO64;
    }
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

    context_attr.field_mask = UCP_ATTR_FIELD_MEMORY_TYPES;
    status = ucp_context_query(ucp_context, &context_attr);
    if (UCS_OK != status) {
        tl_error(self->super.super.lib,
                 "failed to query supported memory types, %s",
                 ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_worker_create;
    }
    self->ucp_memory_types = context_attr.memory_types;
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

    ucc_status = ucc_mpool_init(
        &self->req_mp, 0,
        ucc_max(sizeof(ucc_tl_ucp_task_t), sizeof(ucc_tl_ucp_schedule_t)), 0,
        UCC_CACHE_LINE_SIZE, 8, UINT_MAX, NULL, params->thread_mode,
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
        ucc_status = UCC_ERR_NO_MESSAGE;
        goto err_thread_mode;
    }

    self->remote_info  = NULL;
    self->n_rinfo_segs = 0;
    self->rkeys        = NULL;
    if (params->params.mask & UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS &&
        params->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB) {
        ucc_status = ucc_tl_ucp_ctx_remote_populate(
            self, params->params.mem_params, params->params.oob);
        if (UCC_OK != ucc_status) {
            tl_error(self->super.super.lib, "failed to gather RMA information");
            goto err_thread_mode;
        }
    }
    if (params->context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB) {
        /* Global ctx mode, we will have ctx_map so can use array for eps */
        self->eps = ucc_calloc(params->context->params.oob.n_oob_eps,
                               sizeof(ucp_ep_h), "ucp_eps");
        if (!self->eps) {
            tl_error(self->super.super.lib,
                     "failed to allocate %zd bytes for ucp_eps",
                     params->context->params.oob.n_oob_eps * sizeof(ucp_ep_h));
            ucc_status = UCC_ERR_NO_MEMORY;
            goto err_thread_mode;
        }
    } else {
        self->eps     = NULL;
        self->ep_hash = kh_init(tl_ucp_ep_hash);
    }
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
    char        *rbuf;
    ucc_status_t status;
    char         sbuf;
    void        *req;

    if (ucc_unlikely(oob->n_oob_eps < 2)) {
        return;
    }

    rbuf = ucc_malloc(sizeof(char) * oob->n_oob_eps, "tmp_barrier");
    if (!rbuf) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tmp barrier array",
                 sizeof(char) * oob->n_oob_eps);
        return;
    }
    if (UCC_OK == oob->allgather(&sbuf, rbuf, sizeof(char), oob->coll_info,
                                 &req)) {
        ucc_assert(req);
        while (UCC_OK != (status = oob->req_test(req))) {
            ucp_worker_progress(ctx->ucp_worker);
            if (status < 0) {
                tl_error(ctx->super.super.lib, "failed to test oob req");
                break;
            }
        }
        oob->req_free(req);
    }
    ucc_free(rbuf);
}

ucc_status_t ucc_tl_ucp_rinfo_destroy(ucc_tl_ucp_context_t *ctx)
{
    ucc_rank_t size = UCC_TL_CTX_OOB(ctx).n_oob_eps;
    int        i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < ctx->n_rinfo_segs; j++) {
            if (UCC_TL_UCP_REMOTE_RKEY(ctx, i, j)) {
                ucp_rkey_destroy(UCC_TL_UCP_REMOTE_RKEY(ctx, i, j));
            }
        }
    }
    for (i = 0; i < ctx->n_rinfo_segs; i++) {
        if (ctx->remote_info[i].mem_h) {
            ucp_mem_unmap(ctx->ucp_context, ctx->remote_info[i].mem_h);
        }
        if (ctx->remote_info[i].packed_key) {
            ucp_rkey_buffer_release(ctx->remote_info[i].packed_key);
        }
    }
    ucc_free(ctx->remote_info);
    ucc_free(ctx->rkeys);
    ctx->remote_info = NULL;
    ctx->rkeys       = NULL;

    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_ucp_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_tl_ucp_close_eps(self);
    if (self->eps) {
        ucc_free(self->eps);
    } else {
        kh_destroy(tl_ucp_ep_hash, self->ep_hash);
    }
    if (self->remote_info) {
        ucc_tl_ucp_rinfo_destroy(self);
    }
    if (UCC_TL_CTX_HAS_OOB(self)) {
        ucc_tl_ucp_context_barrier(self, &UCC_TL_CTX_OOB(self));
    }
    ucc_context_progress_deregister(
        self->super.super.ucc_context,
        (ucc_context_progress_fn_t)ucp_worker_progress, self->ucp_worker);
    if (self->worker_address) {
        ucp_worker_release_address(self->ucp_worker, self->worker_address);
    }
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
    if (ucc_unlikely(status != UCS_OK)) {
        return ucs_status_to_ucc_status(status);
    }

    status = ucp_mem_unmap(ctx->ucp_context, mh);
    if (ucc_unlikely(status != UCS_OK)) {
        return ucs_status_to_ucc_status(status);
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_ctx_remote_populate(ucc_tl_ucp_context_t * ctx,
                                            ucc_mem_map_params_t   map,
                                            ucc_context_oob_coll_t oob)
{
    uint32_t             size  = oob.n_oob_eps;
    uint64_t             nsegs = map.n_segments;
    ucp_mem_map_params_t mmap_params;
    ucp_mem_h            mh;
    ucs_status_t         status;
    ucc_status_t         ucc_status;
    int                  i;

    if (size < 2) {
        tl_error(
            ctx->super.super.lib,
            "oob.n_oob_eps set to incorrect value for remote exchange (%d)",
            size);
        return UCC_ERR_INVALID_PARAM;
    }
    if (nsegs > MAX_NR_SEGMENTS) {
        tl_error(ctx->super.super.lib, "cannot map more than %d segments",
                 MAX_NR_SEGMENTS);
        return UCC_ERR_INVALID_PARAM;
    }
    ctx->rkeys =
        (ucp_rkey_h *)ucc_calloc(sizeof(ucp_rkey_h), nsegs * size, "ucp_ctx_rkeys");
    if (NULL == ctx->rkeys) {
        tl_error(ctx->super.super.lib, "failed to allocated %zu bytes",
                 sizeof(ucp_rkey_h) * nsegs * size);
        return UCC_ERR_NO_MEMORY;
    }
    ctx->remote_info = (ucc_tl_ucp_remote_info_t *)ucc_calloc(
        nsegs, sizeof(ucc_tl_ucp_remote_info_t), "ucp_remote_info");
    if (NULL == ctx->remote_info) {
        tl_error(ctx->super.super.lib, "failed to allocated %zu bytes",
                 sizeof(ucc_tl_ucp_remote_info_t) * nsegs);
        ucc_status = UCC_ERR_NO_MEMORY;
        goto fail_alloc_remote_segs;
    }

    for (i = 0; i < nsegs; i++) {
        mmap_params.field_mask =
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mmap_params.address = map.segments[i].address;
        mmap_params.length  = map.segments[i].len;

        status = ucp_mem_map(ctx->ucp_context, &mmap_params, &mh);
        if (UCS_OK != status) {
            tl_error(ctx->super.super.lib,
                     "ucp_mem_map failed with error code: %d", status);
            ucc_status = ucs_status_to_ucc_status(status);
            goto fail_mem_map;
        }
        ctx->remote_info[i].mem_h = (void *)mh;
        status =
            ucp_rkey_pack(ctx->ucp_context, mh, &ctx->remote_info[i].packed_key,
                          &ctx->remote_info[i].packed_key_len);
        if (UCS_OK != status) {
            tl_error(ctx->super.super.lib,
                     "failed to pack UCP key with error code: %d", status);
            ucc_status = ucs_status_to_ucc_status(status);
            goto fail_mem_map;
        }
        ctx->remote_info[i].va_base = map.segments[i].address;
        ctx->remote_info[i].len     = map.segments[i].len;
    }
    ctx->n_rinfo_segs = nsegs;

    return UCC_OK;
fail_mem_map:
    for (i = 0; i < nsegs; i++) {
        if (ctx->remote_info[i].mem_h) {
            ucp_mem_unmap(ctx->ucp_context, ctx->remote_info[i].mem_h);
        }
        if (ctx->remote_info[i].packed_key) {
            ucp_rkey_buffer_release(ctx->remote_info[i].packed_key);
        }
    }
fail_alloc_remote_segs:
    ucc_free(ctx->remote_info);
    ucc_free(ctx->rkeys);
    return ucc_status;
}

static void ucc_tl_ucp_ctx_remote_pack_data(ucc_tl_ucp_context_t *ctx,
                                            void                 *pack)
{
    uint64_t  nsegs          = ctx->n_rinfo_segs;
    uint64_t  offset         = 0;
    size_t    section_offset = sizeof(uint64_t) * nsegs;
    void     *keys;
    uint64_t *rvas;
    uint64_t *lens;
    uint64_t *key_sizes;
    int       i;

    /* pack into one data object in following order: */
    /* rva, len, pack sizes, packed keys */
    rvas      = pack;
    lens      = PTR_OFFSET(pack, section_offset);
    key_sizes = PTR_OFFSET(pack, (section_offset * 2));
    keys      = PTR_OFFSET(pack, (section_offset * 3));

    for (i = 0; i < nsegs; i++) {
        rvas[i]      = (uint64_t)ctx->remote_info[i].va_base;
        lens[i]      = ctx->remote_info[i].len;
        key_sizes[i] = ctx->remote_info[i].packed_key_len;
        memcpy(PTR_OFFSET(keys, offset), ctx->remote_info[i].packed_key,
               ctx->remote_info[i].packed_key_len);
        offset += ctx->remote_info[i].packed_key_len;
    }
}

ucc_status_t ucc_tl_ucp_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t      *attr)
{
    ucc_tl_ucp_context_t *ctx = ucc_derived_of(context, ucc_tl_ucp_context_t);
    ucs_status_t          ucs_status;
    size_t                packed_length;
    int                   i;

    if ((attr->attr.mask & (UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN |
                            UCC_CONTEXT_ATTR_FIELD_CTX_ADDR)) &&
        (NULL == ctx->worker_address)) {
        ucs_status = ucp_worker_get_address(
            ctx->ucp_worker, &ctx->worker_address, &ctx->ucp_addrlen);
        if (UCS_OK != ucs_status) {
            tl_error(ctx->super.super.lib, "failed to get ucp worker address");
            return ucs_status_to_ucc_status(ucs_status);
        }
    }

    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        packed_length = TL_UCP_EP_ADDRLEN_SIZE + ctx->ucp_addrlen;
        if (NULL != ctx->remote_info) {
            packed_length += ctx->n_rinfo_segs * (sizeof(size_t) * 3);
            for (i = 0; i < ctx->n_rinfo_segs; i++) {
                packed_length += ctx->remote_info[i].packed_key_len;
            }
        }
        attr->attr.ctx_addr_len = packed_length;
    }
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR) {
        *((uint64_t*)attr->attr.ctx_addr) = ctx->ucp_addrlen;
        memcpy(TL_UCP_EP_ADDR_WORKER(attr->attr.ctx_addr),
               ctx->worker_address, ctx->ucp_addrlen);
        if (NULL != ctx->remote_info) {
            ucc_tl_ucp_ctx_remote_pack_data(ctx,
                            TL_UCP_EP_ADDR_ONESIDED_INFO(attr->attr.ctx_addr));
        }
    }
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE) {
        attr->attr.global_work_buffer_size =
            ONESIDED_SYNC_SIZE + ONESIDED_REDUCE_SIZE;
    }
    attr->topo_required = 0;
    return UCC_OK;
}
