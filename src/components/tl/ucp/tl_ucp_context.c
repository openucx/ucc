/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_tag.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_ep.h"
#include "utils/ucc_math.h"
#include "schedule/ucc_schedule_pipelined.h"
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
    status = ucp_config_read(params->prefix, NULL, &ucp_config);
    if (UCS_OK != status) {
        tl_error(self->super.super.lib, "failed to read ucp configuration, %s",
                 ucs_status_string(status));
        ucc_status = ucs_status_to_ucc_status(status);
        goto err_cfg;
    }

    ucp_params.field_mask =
        UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_TAG_SENDER_MASK;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA |
                          UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64;
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

    ucc_status = ucc_mpool_init(
        &self->req_mp, 0,
        ucc_max(sizeof(ucc_tl_ucp_task_t), sizeof(ucc_schedule_pipelined_t)), 0,
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

    self->rinfo_hash = NULL;
    self->remote_info = NULL;
    self->n_rinfo_segs = 0;
    if (params->params.mask & UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS &&
        params->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB) {
        ucc_status_t mm_status;

        mm_status = ucc_tl_ucp_ctx_remote_populate(
            self, params->params.mem_params, params->params.oob);
        if (UCC_OK != mm_status) {
            return mm_status;
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
    char        *rbuf = ucc_malloc(sizeof(char) * oob->n_oob_eps,
                                   "tmp_barrier");
    ucc_status_t status;
    char         sbuf;
    void        *req;

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
    ucc_tl_ucp_remote_info_t **rinfo;

    rinfo = (ucc_tl_ucp_remote_info_t **)tl_ucp_hash_rinfo_pop(ctx->rinfo_hash);
    while (rinfo) {
        for (int i = 0; i < ctx->n_rinfo_segs; i++) {
            if (rinfo[0][i].rkey) {
                ucp_rkey_destroy(rinfo[0][i].rkey);
            }
            if (rinfo[0][i].mem_h) {
                ucp_mem_unmap(ctx->ucp_context, rinfo[0][i].mem_h);
            }
            if (rinfo[0][i].packed_key) {
                free(rinfo[0][i].packed_key);
            }
        }
        free(rinfo[0]);
        rinfo =
            (ucc_tl_ucp_remote_info_t **)tl_ucp_hash_rinfo_pop(ctx->rinfo_hash);
    }

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
    if (self->rinfo_hash) {
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

static size_t ucc_tl_ucp_ctx_remote_pack(ucc_mem_map_params_t map,
                                         uint64_t             max_segs,
                                         uint64_t max_pack_size, uint32_t rank,
                                         uint32_t size, void **my_pack,
                                         size_t *my_pack_sizes, void **pack)
{
    void *    packed_data;
    void *    base;
    void *    keys;
    uint64_t *rvas;
    uint64_t *lens;
    uint64_t *key_sizes;
    uint64_t  nsegs          = map.n_maps;
    uint64_t  offset         = 0;
    size_t    total          = 0;
    size_t    section_offset = sizeof(uint64_t) * max_segs;

    // pack the following data :
    // rva, len, pack sizes, packed keys into one data object
    total       = (sizeof(uint64_t) * 3 + max_pack_size) * max_segs;
    packed_data = calloc(total, size);

    base      = packed_data + rank * total;
    rvas      = base;
    lens      = base + section_offset;
    key_sizes = base + (section_offset * 2);
    keys      = base + (section_offset * 3);

    for (int i = 0; i < max_segs; i++) {
        if (i < nsegs) {
            rvas[i]      = (uint64_t)map.maps[i].address;
            lens[i]      = map.maps[i].len;
            key_sizes[i] = my_pack_sizes[i];
            memcpy(keys + offset, my_pack[i], my_pack_sizes[i]);
            offset += my_pack_sizes[i];
        }
        else {
            rvas[i]      = 0;
            lens[i]      = 0;
            key_sizes[i] = 0;
        }
    }

    *pack = packed_data;
    return total;
}

static ucc_status_t ucc_tl_ucp_ctx_exchange_data(void *sbuf, void *rbuf,
                                                 size_t                msg_size,
                                                 ucc_tl_ucp_context_t *ctx,
                                                 ucc_team_oob_coll_t   oob)
{
    void *       req;
    ucc_status_t ucc_status;

    ucc_status = oob.allgather(sbuf, rbuf, msg_size, oob.coll_info, &req);
    if (UCC_OK != ucc_status) {
        tl_error(ctx->super.super.lib,
                 "oob.allgather failed with error code: %d", ucc_status);

        return UCC_ERR_NO_MESSAGE;
    }

    while (UCC_INPROGRESS == (ucc_status = oob.req_test(req)))
        ;
    if (ucc_status < 0) {
        tl_error(ctx->super.super.lib,
                 "oob.allgather failed with error code: %d", ucc_status);

        return UCC_ERR_NO_MESSAGE;
    }
    oob.req_free(req);

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_ctx_remote_populate(ucc_tl_ucp_context_t *ctx,
                                            ucc_mem_map_params_t  map,
                                            ucc_team_oob_coll_t   oob)
{
    ucc_tl_ucp_remote_info_t **remote_info;
    ucp_mem_map_params_t       mmap_params;
    ucp_mem_h                  mh;
    ucs_status_t               status;
    ucc_status_t               ucc_status;
    uint32_t                   rank  = oob.oob_ep;
    uint32_t                   size  = oob.n_oob_eps;
    uint64_t                   nsegs = map.n_maps;
    uint64_t                   rsegs[size * 2]; // remote segments & total size
    uint64_t                   send_rsegs[2];
    void *                     packed_data; // all data to exchange
    void *                     my_pack[nsegs];
    size_t                     total    = 0;
    size_t                     max_segs = nsegs;
    size_t                     max_pack_size;
    size_t                     my_pack_sizes[nsegs];

    ctx->rinfo_hash = kh_init(tl_ucp_rinfo_hash);
    remote_info     = (ucc_tl_ucp_remote_info_t **)malloc(
        sizeof(ucc_tl_ucp_remote_info_t *) * size);

    // local setup
    remote_info[rank] = (ucc_tl_ucp_remote_info_t *)calloc(
        nsegs, sizeof(ucc_tl_ucp_remote_info_t));

    for (int i = 0; i < nsegs; i++) {
        void *addr = map.maps[i].address;

        // TODO: perform allocation based on hints/constraints if addr NULL

        mmap_params.field_mask =
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mmap_params.address = addr;
        mmap_params.length  = map.maps[i].len;

        status = ucp_mem_map(ctx->ucp_context, &mmap_params, &mh);
        remote_info[rank][i].mem_h = (void *)mh;

        // pack our key here
        status =
            ucp_rkey_pack(ctx->ucp_context, mh, &my_pack[i], &my_pack_sizes[i]);
        if (UCS_OK != status) {
            tl_error(ctx->super.super.lib,
                     "failed to pack UCP key with error code: %d", status);
            return UCC_ERR_NO_MESSAGE;
        }

        total += my_pack_sizes[i];
    }

    // exchange number of segments
    send_rsegs[0] = nsegs;
    send_rsegs[1] = total;
    ucc_status    = ucc_tl_ucp_ctx_exchange_data(send_rsegs, rsegs,
                                              sizeof(uint64_t) * 2, ctx, oob);
    if (UCC_OK != ucc_status) {
        return ucc_status;
    }

    // calc maximum number of segments and UCP packed key size
    max_pack_size = total;
    for (int i = 0, k = 0; i < size; i++, k += 2) {
        if (rsegs[k] > max_segs) {
            max_segs = rsegs[k];
        }
        if (rsegs[k + 1] > max_pack_size) {
            max_pack_size = rsegs[k + 1];
        }
    }

    // pack all data to be exchanged
    total = ucc_tl_ucp_ctx_remote_pack(map, max_segs, max_pack_size, rank, size,
                                       my_pack, my_pack_sizes, &packed_data);

    // exchange info
    ucc_status = ucc_tl_ucp_ctx_exchange_data(packed_data + rank * total,
                                              packed_data, total, ctx, oob);
    if (UCC_OK != ucc_status) {
        return ucc_status;
    }

    // unpack the exchanged data
    for (int i = 0, k = 0; i < size; i++, k += 2) {
        void *    base      = packed_data + i * total;
        uint64_t *rvas      = base;
        uint64_t *lens      = base + sizeof(uint64_t) * max_segs;
        uint64_t *key_sizes = base + (sizeof(uint64_t) * max_segs * 2);
        void *    key       = base + (sizeof(uint64_t) * max_segs) * 3;
        size_t    offset    = 0;

        if (i != rank) {
            remote_info[i] = (ucc_tl_ucp_remote_info_t *)calloc(
                rsegs[k], sizeof(ucc_tl_ucp_remote_info_t));
        }

        for (int j = 0; j < max_segs; j++) {
            if (j < rsegs[k]) {
               remote_info[i][j].va_base    = (void *)rvas[j];
               remote_info[i][j].len        = lens[j];
               remote_info[i][j].packed_key = malloc(key_sizes[j]);
               memcpy(remote_info[i][j].packed_key, key + offset,
                      key_sizes[j]);
               offset += key_sizes[j];
            } else {
               remote_info[i][j].va_base = 0;
               remote_info[i][j].len     = 0;
            }
        }
    }

    free(packed_data);
    ctx->remote_info  = remote_info;
    ctx->n_rinfo_segs = nsegs;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_get_context_attr(const ucc_base_context_t *context,
                                         ucc_base_ctx_attr_t      *attr)
{
    ucc_tl_ucp_context_t *ctx = ucc_derived_of(context, ucc_tl_ucp_context_t);
    ucs_status_t          ucs_status;
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
        attr->attr.ctx_addr_len = ctx->ucp_addrlen;
    }
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR) {
        memcpy(attr->attr.ctx_addr, ctx->worker_address, ctx->ucp_addrlen);
    }
    attr->topo_required = 0;
    return UCC_OK;
}
