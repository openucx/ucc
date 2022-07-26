/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <inttypes.h>
#include "tl_sharp.h"
#include "utils/arch/cpu.h"

static ucc_mpool_ops_t ucc_tl_sharp_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static int ucc_tl_sharp_oob_barrier(void *arg)
{
    ucc_tl_sharp_oob_ctx_t *oob_ctx  = (ucc_tl_sharp_oob_ctx_t *)arg;
    ucc_tl_sharp_context_t *ctx      = oob_ctx->ctx;
    ucc_oob_coll_t         *oob_coll = oob_ctx->oob;
    ucc_status_t            status;
    char                    sbuf, *rbuf;
    void                   *req;

    rbuf = ucc_malloc(sizeof(char) * oob_coll->n_oob_eps, "tmp_barrier");
    if (!rbuf) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tmp barrier array",
                 sizeof(char) * oob_coll->n_oob_eps);
        return UCC_ERR_NO_MEMORY;
    }

    status = oob_coll->allgather(&sbuf, rbuf, sizeof(char),
                                 oob_coll->coll_info, &req);
    if (UCC_OK == status) {
        ucc_assert(req);
        while (UCC_OK != (status = oob_coll->req_test(req))) {
            if (status < 0) {
                tl_error(ctx->super.super.lib, "failed to test oob req");
                break;
            }
        }
        oob_coll->req_free(req);
    }

    ucc_free(rbuf);
    return status;
}

static int ucc_tl_sharp_oob_gather(void *arg, int root, void *sbuf,
                                   void *rbuf, int size)
{
    ucc_tl_sharp_oob_ctx_t *oob_ctx  = (ucc_tl_sharp_oob_ctx_t *)arg;
    ucc_tl_sharp_context_t *ctx      = oob_ctx->ctx;
    void                   *tmp_rbuf = NULL;
    size_t                  msg_size = size;
    ucc_oob_coll_t         *oob_coll = oob_ctx->oob;
    void                   *req;
    ucc_status_t            status;

    if (oob_coll->oob_ep != root) {
        tmp_rbuf = ucc_malloc(msg_size * oob_coll->n_oob_eps, "tmp_gather");
        if (!tmp_rbuf) {
            tl_error(ctx->super.super.lib,
                     "failed to allocate %zd bytes for tmp barrier array",
                     msg_size * oob_coll->n_oob_eps);
            return UCC_ERR_NO_MEMORY;
        }
        rbuf = tmp_rbuf;
    }

    status = oob_coll->allgather(sbuf, rbuf, msg_size, oob_coll->coll_info, &req);
    if (UCC_OK == status) {
        ucc_assert(req);
        while (UCC_OK != (status = oob_coll->req_test(req))) {
            if (status < 0) {
                tl_error(ctx->super.super.lib, "failed to test oob req");
                break;
            }
        }

        oob_coll->req_free(req);
    }

    if (tmp_rbuf) {
        ucc_free(tmp_rbuf);
    }
    return status;
}

static int ucc_tl_sharp_oob_bcast(void *arg, void *buf, int size, int root)
{
    ucc_tl_sharp_oob_ctx_t *oob_ctx  = (ucc_tl_sharp_oob_ctx_t *)arg;
    ucc_tl_sharp_context_t *ctx      = oob_ctx->ctx;
    size_t                  msg_size = size;
    ucc_oob_coll_t         *oob_coll = oob_ctx->oob;
    ucc_status_t            status;
    void                   *req, *tmp_rbuf;

    tmp_rbuf = ucc_malloc(msg_size * oob_coll->n_oob_eps, "tmp_barrier");
    if (!tmp_rbuf) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for tmp barrier array",
                 msg_size * oob_coll->n_oob_eps);
        return UCC_ERR_NO_MEMORY;
    }

    status = oob_coll->allgather(buf, tmp_rbuf, msg_size, oob_coll ->coll_info,
                                 &req);
    if (UCC_OK == status) {
        ucc_assert(req);
        while (UCC_OK != (status = oob_coll ->req_test(req))) {
            if (status < 0) {
                tl_error(ctx->super.super.lib, "failed to test oob req");
                break;
            }
        }
        oob_coll ->req_free(req);
    }

    memcpy(buf, PTR_OFFSET(tmp_rbuf, root * msg_size), msg_size);

    ucc_free(tmp_rbuf);
    return status;
}

static ucs_status_t
ucc_tl_sharp_rcache_mem_reg_cb(void *context, ucc_rcache_t *rcache,
                               void *arg, ucc_rcache_region_t *rregion,
                               uint16_t flags)
{
    ucc_tl_sharp_context_t       *ctx = (ucc_tl_sharp_context_t*) context;
    ucc_tl_sharp_rcache_region_t *region;
    void                         *address;
    size_t                        length;
    int                           ret;

    address = (void*)rregion->super.start;
    length  = (size_t)(rregion->super.end - rregion->super.start);
    region  = ucc_derived_of(rregion, ucc_tl_sharp_rcache_region_t);

    ret = sharp_coll_reg_mr(ctx->sharp_context, address,
                            length, &region->reg.mr);
    if (ret < 0) {
        tl_error(ctx->super.super.lib, "reg failed(%d). addr:%p len:%zd",
                 ret, address, length);
        return UCS_ERR_INVALID_PARAM;
    } else {
        tl_debug(ctx->super.super.lib, "region:%p reg mr:%p addr:%p len:%zd",
                 rregion, region->reg.mr, address, length);
        return UCS_OK;
    }
}

static void ucc_tl_sharp_rcache_mem_dereg_cb(void *context, ucc_rcache_t *rcache,
                                             ucc_rcache_region_t *rregion)
{
    ucc_tl_sharp_context_t       *ctx    = (ucc_tl_sharp_context_t*)context;
    ucc_tl_sharp_rcache_region_t *region = ucc_derived_of(rregion,
                                                ucc_tl_sharp_rcache_region_t);
    int                           ret;

    ret = sharp_coll_dereg_mr(ctx->sharp_context, region->reg.mr);
    if (ret < 0) {
        tl_error(ctx->super.super.lib, "dereg failed(%d). mr:%p",
                 ret, region->reg.mr);
    } else {
        tl_debug(ctx->super.super.lib, "rregion:%p dereg mr:%p",
                 rregion, region->reg.mr);
    }
}

static void
ucc_tl_sharp_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                   ucs_rcache_region_t *rregion, char *buf,
                                   size_t max)
{
    ucc_tl_sharp_rcache_region_t *region = ucc_derived_of(rregion,
                                           ucc_tl_sharp_rcache_region_t);

    snprintf(buf, max, "bar ptr:%p", region->reg.mr);
}

static ucc_rcache_ops_t ucc_tl_sharp_rcache_ops = {
    .mem_reg     = ucc_tl_sharp_rcache_mem_reg_cb,
    .mem_dereg   = ucc_tl_sharp_rcache_mem_dereg_cb,
    .dump_region = ucc_tl_sharp_rcache_dump_region_cb
};

UCC_CLASS_INIT_FUNC(ucc_tl_sharp_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_sharp_context_config_t *tl_sharp_config =
        ucc_derived_of(config, ucc_tl_sharp_context_config_t);
    struct sharp_coll_init_spec    init_spec = {0};
    ucc_status_t                   status;
    ucc_rcache_params_t            rcache_params;
    struct timeval                 tval;

    if (!(params->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB)) {
        tl_error(tl_sharp_config->super.tl_lib, "Context OOB is required for SHARP");
        status = UCC_ERR_INVALID_PARAM;
        goto err;
    }

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_sharp_config->super,
                              params->context);
    memcpy(&self->cfg, tl_sharp_config, sizeof(*tl_sharp_config));

    if (self->cfg.rand_seed == 0) {
        gettimeofday(&tval, NULL);
        self->cfg.rand_seed = (int) tval.tv_usec;
    }

    self->rcache       = NULL;
    self->oob_ctx.ctx  = self;
    self->oob_ctx.oob  = &UCC_TL_CTX_OOB(self);

    init_spec.progress_func                  = NULL;
    init_spec.world_rank                     = UCC_TL_CTX_OOB(self).oob_ep;
    init_spec.world_local_rank               = 0;
    init_spec.world_size                     = UCC_TL_CTX_OOB(self).n_oob_eps;
    init_spec.enable_thread_support          =
                    (params->thread_mode == UCC_THREAD_MULTIPLE) ? 1 : 0;
    init_spec.group_channel_idx              = 0;
    init_spec.oob_colls.barrier              = ucc_tl_sharp_oob_barrier;
    init_spec.oob_colls.bcast                = ucc_tl_sharp_oob_bcast;
    init_spec.oob_colls.gather               = ucc_tl_sharp_oob_gather;
    init_spec.oob_ctx                        = &self->oob_ctx;
    init_spec.config                         = sharp_coll_default_config;
    init_spec.config.user_progress_num_polls = self->cfg.uprogress_num_polls;
    init_spec.config.ib_dev_list             = self->cfg.dev_list;
    init_spec.job_id                         = ((getpid() ^ pthread_self())
                                                ^ rand_r(&self->cfg.rand_seed));

    //TODO: replace with unique context ID?
    ucc_tl_sharp_oob_bcast((void *)&self->oob_ctx,  &init_spec.job_id,
                           sizeof(uint64_t), 0);

    int ret = sharp_coll_init(&init_spec, &self->sharp_context);
    if (ret < 0 ) {
        tl_debug(self->super.super.lib, "Failed to initialize SHARP "
                 "collectives:%s(%d) job ID:%" PRIu64"\n",
                 sharp_coll_strerror(ret), ret, init_spec.job_id);
        status = UCC_ERR_NO_RESOURCE;
        goto err;
    }

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_sharp_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_tl_sharp_req_mpool_ops, params->thread_mode,
                            "tl_sharp_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_sharp_req mpool");
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }

   if (UCC_OK != ucc_context_progress_register(
                      params->context,
                      (ucc_context_progress_fn_t)sharp_coll_progress,
                      self->sharp_context)) {
        tl_error(self->super.super.lib, "failed to register progress function");
        status = UCC_ERR_NO_MESSAGE;
        goto err_clean_mpool;
   }

   if (self->cfg.use_rcache) {
       rcache_params.alignment          = 64;
       rcache_params.ucm_event_priority = 1000;
       rcache_params.max_regions        = ULONG_MAX;
       rcache_params.max_size           = SIZE_MAX;
       rcache_params.region_struct_size = sizeof(ucc_tl_sharp_rcache_region_t);
       rcache_params.max_alignment      = getpagesize();
       rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED |
                                          UCM_EVENT_MEM_TYPE_FREE;
       rcache_params.context            = self;
       rcache_params.ops                = &ucc_tl_sharp_rcache_ops;
       rcache_params.flags              = 0;

       status = ucc_rcache_create(&rcache_params, "SHARP", &self->rcache);
       if (status != UCC_OK) {
           tl_error(self->super.super.lib, "failed to create rcache");
           status = UCC_ERR_NO_RESOURCE;
           goto err_clean_progress;
       }
   }

   tl_info(self->super.super.lib, "initialized tl context: %p", self);
   return UCC_OK;

err_clean_progress:
    ucc_context_progress_deregister(
        params->context, (ucc_context_progress_fn_t)sharp_coll_progress,
        self->sharp_context);
err_clean_mpool:
    ucc_mpool_cleanup(&self->req_mp, 0);
err:
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_sharp_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);

    if (self->rcache != NULL) {
        ucc_rcache_destroy(self->rcache);
    }

    ucc_context_progress_deregister(
        self->super.super.ucc_context,
        (ucc_context_progress_fn_t)sharp_coll_progress, self->sharp_context);
    sharp_coll_finalize(self->sharp_context);
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_sharp_context_t, ucc_tl_context_t);

ucc_status_t ucc_tl_sharp_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                                           ucc_base_ctx_attr_t *attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }

    attr->topo_required = 0;

    return UCC_OK;
}
