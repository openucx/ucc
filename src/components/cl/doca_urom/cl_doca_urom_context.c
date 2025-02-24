/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_doca_urom.h"
#include "cl_doca_urom_coll.h"
#include "utils/ucc_malloc.h"

static doca_error_t ucc_status_to_doca_error(ucc_status_t status)
{
    doca_error_t doca_err = DOCA_ERROR_UNKNOWN;

    switch (status) {
    case UCC_OK:
        doca_err = DOCA_SUCCESS;
        break;
    case UCC_INPROGRESS:
    case UCC_OPERATION_INITIALIZED:
        doca_err = DOCA_ERROR_IN_PROGRESS;
        break;
    case UCC_ERR_NOT_SUPPORTED:
    case UCC_ERR_NOT_IMPLEMENTED:
        doca_err = DOCA_ERROR_NOT_SUPPORTED;
        break;
    case UCC_ERR_INVALID_PARAM:
        doca_err = DOCA_ERROR_INVALID_VALUE;
        break;
    case UCC_ERR_NO_MEMORY:
        doca_err = DOCA_ERROR_NO_MEMORY;
        break;
    case UCC_ERR_NO_RESOURCE:
        doca_err = DOCA_ERROR_FULL;
        break;
    case UCC_ERR_NO_MESSAGE:
    case UCC_ERR_LAST:
        doca_err = DOCA_ERROR_UNKNOWN;
        break;
    case UCC_ERR_NOT_FOUND:
        doca_err = DOCA_ERROR_NOT_FOUND;
        break;
    case UCC_ERR_TIMED_OUT:
        doca_err = DOCA_ERROR_TIME_OUT;
        break;
    }

    return doca_err;
}

// Convert the ucc oob allgather test to work with doca_error_t.
// The problem this solves is that DOCA_ERROR_IN_PROGRESS is numerically
// equivalent to 26 while UCC_INPROGRESS is equal to 1
ucc_status_t (*params_oob_allgather_test)(void *req);
static doca_error_t oob_allgather_test_docafied(void *req)
{
    return ucc_status_to_doca_error(params_oob_allgather_test(req));
}

ucc_status_t (*params_oob_allgather_free)(void *req);
static doca_error_t oob_allgather_free_docafied(void *req)
{
    return ucc_status_to_doca_error(params_oob_allgather_free(req));
}

ucc_status_t (*params_oob_allgather)(void *, void *, size_t, void *, void **);
static doca_error_t oob_allgather_docafied(void * s, void * r, size_t z,
                                           void * i, void **req_p)
{
    return ucc_status_to_doca_error(params_oob_allgather(s,r,z,i,req_p));
}

UCC_CLASS_INIT_FUNC(ucc_cl_doca_urom_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t         *config)
{
    struct ucc_cl_doca_urom_domain_buffer_attrs buf_attrs     = {0};
    struct doca_urom_domain_oob_coll            oob_coll      = {0};
    doca_error_t                                tmp_result    = DOCA_SUCCESS;
    union doca_data                             cookie        = {0};
    struct ucc_cl_doca_urom_result              res           = {0};
    doca_error_t                                result        = DOCA_SUCCESS;
    size_t                                      length        = 4096;
    int                                         ucp_index     = -1;
    int                                         num_envs      = 0;
    char                                      **envs          = NULL;
    size_t                                      plugins_count = 0;
    struct doca_log_backend                    *sdk_log       = NULL;
    const ucc_cl_doca_urom_context_config_t    *cl_config     =
        ucc_derived_of(config, ucc_cl_doca_urom_context_config_t);
    ucc_cl_doca_urom_lib_t                     *doca_urom_lib =
        ucc_derived_of(cl_config->super.cl_lib, ucc_cl_doca_urom_lib_t);
    ucc_config_names_array_t                   *tls           =
        &cl_config->super.cl_lib->tls.array;
    ucc_lib_params_t                            lib_params    = {
        .mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    const struct doca_urom_service_plugin_info *plugins;
    ucc_tl_ucp_context_t                       *tl_ctx;
    enum doca_ctx_states                        state;
    struct export_buf                           ebuf;
    ucc_status_t                                status;
    ucs_status_t                                ucs_status;
    ucc_rank_t                                  rank;
    uint64_t                                    rank_u64;
    size_t                                      i;
    void                                       *buffer;
    int                                         ret;
    char                                       *plugin_name;
    char                                       *device;

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_context_t, &cl_config->super,
                              params->context);
    memcpy(&self->cfg, cl_config, sizeof(*cl_config));

    if (tls->count == 1 && !strcmp(tls->names[0], "all")) {
        tls = &params->context->all_tls;
    }
    self->super.tl_ctxs = ucc_malloc(sizeof(ucc_tl_context_t*) * tls->count,
                            "cl_doca_urom_tl_ctxs");
    if (!self->super.tl_ctxs) {
        cl_error(cl_config->super.cl_lib,
                 "failed to allocate %zd bytes for tl_ctxs",
                 sizeof(ucc_tl_context_t**) * tls->count);
        return UCC_ERR_NO_MEMORY;
    }
    self->super.n_tl_ctxs = 0;
    for (i = 0; i < tls->count; i++) {
        ucc_debug("TL NAME[%zu]: %s", i, tls->names[i]);
        if (strcmp(tls->names[i], "ucp") == 0) {
            status = ucc_tl_context_get(params->context, tls->names[i],
                                   &self->super.tl_ctxs[self->super.n_tl_ctxs]);
            if (UCC_OK != status) {
                cl_error(cl_config->super.cl_lib, "TL ucp not available");
            } 
            ucp_index = self->super.n_tl_ctxs;
            doca_urom_lib->tl_ucp_index = ucp_index;
            self->super.n_tl_ctxs++;
        }
    }
    if (0 == self->super.n_tl_ctxs) {
        cl_error(cl_config->super.cl_lib, "no TL contexts are available");
        ucc_free(self->super.tl_ctxs);
        self->super.tl_ctxs = NULL;
        return UCC_ERR_NOT_FOUND;
    }

    ucc_assert(ucp_index != -1);
    tl_ctx = ucc_derived_of(self->super.tl_ctxs[ucp_index],
                            ucc_tl_ucp_context_t);

    memset(&self->urom_ctx, 0, sizeof(ucc_cl_doca_urom_ctx_t));

    ucc_assert(params->params.mask | UCC_CONTEXT_PARAM_FIELD_OOB);
    self->urom_ctx.ctx_rank = params->params.oob.oob_ep;
    rank = self->urom_ctx.ctx_rank;

    if (self->cfg.plugin_envs.count > 0) {
        num_envs = self->cfg.plugin_envs.count;
        envs = self->cfg.plugin_envs.names;
    }

    plugin_name = self->cfg.plugin_name;
    device      = self->cfg.device;

    result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib,
                 "failed to create DOCA log backend");
        return UCC_ERR_NO_RESOURCE;
    }
    result = doca_log_backend_set_sdk_level(sdk_log, cl_config->doca_log_level);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to set backend sdk level");
        return UCC_ERR_NO_RESOURCE;
    }

    result = ucc_cl_doca_urom_open_doca_device_with_ibdev_name(
                (uint8_t *)device, strlen(device),
                NULL, &self->urom_ctx.dev);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to open device %s", device);
        return UCC_ERR_NO_RESOURCE;
    }

    result = doca_pe_create(&self->urom_ctx.urom_pe);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to create DOCA PE");
        goto dev_close;
    }

    result = ucc_cl_doca_urom_start_urom_service(
                self->urom_ctx.urom_pe, self->urom_ctx.dev, 2,
                &self->urom_ctx.urom_service);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib,
                 "failed to create UROM service context");
        goto pe_destroy;
    }

    result = doca_urom_service_get_plugins_list(self->urom_ctx.urom_service,
                                                &plugins, &plugins_count);
    if (result != DOCA_SUCCESS || plugins_count == 0) {
        cl_error(cl_config->super.cl_lib,
                 "failed to get UROM plugins list. plugins_count: %ld",
                 plugins_count);
        goto service_stop;
    }

    for (i = 0; i < plugins_count; i++) {
        if (strcmp(plugin_name, plugins[i].plugin_name) == 0) {
            self->urom_ctx.ucc_info = &plugins[i];
            break;
        }
    }

    if (self->urom_ctx.ucc_info == NULL) {
        cl_error(cl_config->super.cl_lib, "failed to match UCC plugin");
        result = DOCA_ERROR_INVALID_VALUE;
        goto service_stop;
    }

    /* Each command requires a plugin id--save it in the worker */
    result = ucc_cl_doca_urom_save_plugin_id(self->urom_ctx.ucc_info->id,
                                             self->urom_ctx.ucc_info->version);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to init UCC worker plugin");
        goto service_stop;
    }

    self->urom_ctx.urom_worker_addr = ucc_calloc(1,
                                                 UCC_CL_DOCA_UROM_ADDR_MAX_LEN,
                                                 "doca_urom worker addr");
    if (!self->urom_ctx.urom_worker_addr) {
        cl_error(cl_config->super.cl_lib, "failed to allocate %d bytes",
                 UCC_CL_DOCA_UROM_ADDR_MAX_LEN);
        return UCC_ERR_NO_MEMORY;
    }

    /* Create and start worker context */
    result = ucc_cl_doca_urom_start_urom_worker(self->urom_ctx.urom_pe,
                self->urom_ctx.urom_service, rank, NULL,
                16, NULL, envs, num_envs,
                self->urom_ctx.ucc_info->id,
                &self->urom_ctx.urom_worker);
    if (result != DOCA_SUCCESS)
        cl_error(cl_config->super.cl_lib, "failed to start urom worker");

    /* Loop till worker state changes to running */
    do {
        doca_pe_progress(self->urom_ctx.urom_pe);
        result = doca_ctx_get_state(
                    doca_urom_worker_as_ctx(self->urom_ctx.urom_worker),
                    &state);
    } while (state == DOCA_CTX_STATE_STARTING && result == DOCA_SUCCESS);
    if (state != DOCA_CTX_STATE_RUNNING || result != DOCA_SUCCESS) {
        goto worker_cleanup;
    }

    /* Start the UROM domain */
    buffer = ucc_calloc(1, length, "doca_urom domain buffer");
    if (buffer == NULL) {
        cl_error(cl_config->super.cl_lib,
                 "failed to allocate urom domain buffer");
        result = DOCA_ERROR_NO_MEMORY;
        goto worker_cleanup;
    }

    params_oob_allgather      = params->params.oob.allgather;
    oob_coll.allgather        = oob_allgather_docafied;
    params_oob_allgather_test = params->params.oob.req_test;
    oob_coll.req_test         = oob_allgather_test_docafied;
    params_oob_allgather_free = params->params.oob.req_free;
    oob_coll.req_free         = oob_allgather_free_docafied;
    oob_coll.coll_info        = params->params.oob.coll_info;
    oob_coll.n_oob_indexes    = params->params.oob.n_oob_eps;
    oob_coll.oob_index        = rank;

    ucs_status = ucp_worker_get_address(tl_ctx->worker.ucp_worker,
                                        &tl_ctx->worker.worker_address,
                                        &tl_ctx->worker.ucp_addrlen);
    if (ucs_status != UCS_OK) {
        cl_error(cl_config->super.cl_lib, "failed to get ucp worker address");
        goto worker_cleanup;
    }

    result = (doca_error_t) ucc_cl_doca_urom_buffer_export_ucc(
                                tl_ctx->worker.ucp_context, buffer,
                                length, &ebuf);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to export buffer");
        goto worker_cleanup;
    }

    /* The buffers in the domain are used for gets/puts from the host without
       XGVMI. Also, the domain is used for the OOB exchange given to the DPU-
       side UCC instance */
    buf_attrs.buffer   = buffer;
    buf_attrs.buf_len  = length;
    buf_attrs.memh     = ebuf.packed_memh;
    buf_attrs.memh_len = ebuf.packed_memh_len;
    buf_attrs.mkey     = ebuf.packed_key;
    buf_attrs.mkey_len = ebuf.packed_key_len;

    /* Create domain context */
    rank_u64 = (uint64_t)rank;
    result = ucc_cl_doca_urom_start_urom_domain(
                self->urom_ctx.urom_pe,
                &oob_coll,
                &rank_u64, &self->urom_ctx.urom_worker,
                1, &buf_attrs, 1,
                &self->urom_ctx.urom_domain);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to start domain");
        goto worker_unmap;
    }

    /* Loop till domain state changes to running */
    do {
        doca_pe_progress(self->urom_ctx.urom_pe);
        result = doca_ctx_get_state(
                    doca_urom_domain_as_ctx(
                        self->urom_ctx.urom_domain),
                    &state);
    } while (state == DOCA_CTX_STATE_STARTING && result == DOCA_SUCCESS);

    if (state != DOCA_CTX_STATE_RUNNING || result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to start domain");
        result = DOCA_ERROR_BAD_STATE;
        goto worker_unmap;
    }

    /* Create lib */
    cookie.ptr = &res;
    result = ucc_cl_doca_urom_task_lib_create(
                self->urom_ctx.urom_worker,
                cookie, rank, &lib_params,
                ucc_cl_doca_urom_lib_create_finished);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to create lib creation task");
        goto domain_stop;
    }
    do {
        ret = doca_pe_progress(self->urom_ctx.urom_pe);
    } while (ret == 0 && res.result == DOCA_SUCCESS);

    if (res.result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to finish lib create task");
        result = res.result;
        goto domain_stop;
    }
    cl_debug(cl_config->super.cl_lib, "UCC lib create is done");

    cl_debug(cl_config->super.cl_lib, "Creating pd channel");
    result = ucc_cl_doca_urom_task_pd_channel(self->urom_ctx.urom_worker,
                cookie,
                rank,
                tl_ctx->worker.worker_address,
                tl_ctx->worker.ucp_addrlen,
                ucc_cl_doca_urom_pss_dc_finished);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to create data channel task");
        goto lib_destroy;
    }

    do {
        ret = doca_pe_progress(self->urom_ctx.urom_pe);
    } while (ret == 0 && res.result == DOCA_SUCCESS);

    if (res.result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "passive data channel task failed");
        result = res.result;
        goto lib_destroy;
    }
    cl_debug(cl_config->super.cl_lib, "passive data channel is done");

    cl_debug(cl_config->super.cl_lib, "creating task ctx");
    result = ucc_cl_doca_urom_task_ctx_create(self->urom_ctx.urom_worker,
                cookie, rank, 0, NULL, 1,
                params->params.oob.n_oob_eps, 0x0,
                length,
                ucc_cl_doca_urom_ctx_create_finished);
    if (result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to create UCC context task");
        goto lib_destroy;
    }

    do {
        ret = doca_pe_progress(self->urom_ctx.urom_pe);
    } while (ret == 0 && res.result == DOCA_SUCCESS);

    if (res.result != DOCA_SUCCESS || res.context_create.context == NULL) {
        cl_error(cl_config->super.cl_lib, "UCC context create task failed");
        result = res.result;
        goto lib_destroy;
    }
    cl_debug(cl_config->super.cl_lib,
             "UCC context create is done, ucc_context: %p",
             res.context_create.context);
    self->urom_ctx.urom_ucc_context = res.context_create.context;

    status = ucc_mpool_init(&self->sched_mp, 0,
                            sizeof(ucc_cl_doca_urom_schedule_t),
                            0, UCC_CACHE_LINE_SIZE, 2, UINT_MAX,
                            &ucc_coll_task_mpool_ops, params->thread_mode,
                            "cl_doca_urom_sched_mp");
    if (UCC_OK != status) {
        cl_error(cl_config->super.cl_lib,
                 "failed to initialize cl_doca_urom_sched mpool");
        goto lib_destroy;
    }

    cl_debug(cl_config->super.cl_lib, "initialized cl context: %p", self);
    return UCC_OK;

lib_destroy:
    result = ucc_cl_doca_urom_task_lib_destroy(self->urom_ctx.urom_worker,
                cookie, rank, ucc_cl_doca_urom_lib_destroy_finished);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib,
                 "failed to create UCC lib destroy task");
    }

    do {
        ret = doca_pe_progress(self->urom_ctx.urom_pe);
    } while (ret == 0 && res.result == DOCA_SUCCESS);

    if (res.result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "UCC lib destroy failed");
        result = res.result;
    }

domain_stop:
    result = doca_ctx_stop(
                doca_urom_domain_as_ctx(self->urom_ctx.urom_domain));
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to stop UROM domain");
    }

    result = doca_urom_domain_destroy(self->urom_ctx.urom_domain);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to destroy UROM domain");
    }

worker_unmap:
    ucs_status = ucp_mem_unmap(tl_ctx->worker.ucp_context, ebuf.memh);
    if (ucs_status != UCS_OK) {
        cl_error(cl_config->super.cl_lib, "failed to unmap memh");
    }
    free(buffer);

worker_cleanup:
    tmp_result = doca_urom_worker_destroy(self->urom_ctx.urom_worker);
    if (tmp_result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to destroy UROM worker");
    }

service_stop:
    tmp_result = doca_ctx_stop(
                    doca_urom_service_as_ctx(self->urom_ctx.urom_service));
    if (tmp_result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to stop UROM service");
    }
    tmp_result = doca_urom_service_destroy(self->urom_ctx.urom_service);
    if (tmp_result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to destroy UROM service");
    }

pe_destroy:
    tmp_result = doca_pe_destroy(self->urom_ctx.urom_pe);
    if (tmp_result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to destroy PE");
    }

dev_close:
    tmp_result = doca_dev_close(self->urom_ctx.dev);
    if (tmp_result != DOCA_SUCCESS) {
        cl_error(cl_config->super.cl_lib, "failed to close device");
    }

    return UCC_ERR_NO_MESSAGE;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_doca_urom_context_t)
{
    struct ucc_cl_doca_urom_result res    = {0};
    union doca_data                cookie = {0};
    doca_error_t                   result = DOCA_SUCCESS;
    ucc_rank_t                     rank;
    int                            i, ret;

    rank = self->urom_ctx.ctx_rank;
    cookie.ptr = &res;

    result = ucc_cl_doca_urom_task_lib_destroy(self->urom_ctx.urom_worker,
                cookie, rank, ucc_cl_doca_urom_lib_destroy_finished);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib,
                 "failed to create UCC lib destroy task");
    }

    do {
        ret = doca_pe_progress(self->urom_ctx.urom_pe);
    } while (ret == 0 && res.result == DOCA_SUCCESS);

    if (res.result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "UCC lib destroy failed");
        result = res.result;
    }

    result = doca_ctx_stop(
                doca_urom_domain_as_ctx(self->urom_ctx.urom_domain));
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to stop UROM domain");
    }

    result = doca_urom_domain_destroy(self->urom_ctx.urom_domain);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to destroy UROM domain");
    }

    result = doca_ctx_stop(
                doca_urom_service_as_ctx(self->urom_ctx.urom_service));
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to stop UROM service");
    }
    result = doca_urom_service_destroy(self->urom_ctx.urom_service);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to destroy UROM service");
    }

    result = doca_pe_destroy(self->urom_ctx.urom_pe);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to destroy PE");
    }

    result = doca_dev_close(self->urom_ctx.dev);
    if (result != DOCA_SUCCESS) {
        cl_error(self->super.super.lib, "failed to close device");
    }

    cl_debug(self->super.super.lib, "finalizing cl context: %p", self);
    for (i = 0; i < self->super.n_tl_ctxs; i++) {
        ucc_tl_context_put(self->super.tl_ctxs[i]);
    }
    ucc_free(self->super.tl_ctxs);
}

UCC_CLASS_DEFINE(ucc_cl_doca_urom_context_t, ucc_cl_context_t);

ucc_status_t
ucc_cl_doca_urom_get_context_attr(const ucc_base_context_t *context,
                                  ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
    return UCC_OK;
}

ucc_status_t ucc_cl_doca_urom_mem_map(const ucc_base_context_t *context,
                                      int type, void *memh, void *tl_h)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_cl_doca_urom_mem_unmap(const ucc_base_context_t *context,
                                        int type, void *tl_h)
{
    return UCC_ERR_NOT_SUPPORTED;
}

ucc_status_t ucc_cl_doca_urom_memh_pack(const ucc_base_context_t *context,
                                        int type, void *memh, void **packed_buffer)
{
    return UCC_ERR_NOT_SUPPORTED;
}
