/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_context.h"
#include "components/cl/ucc_cl.h"
#include "components/tl/ucc_tl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "ucc_progress_queue.h"
ucc_status_t ucc_context_config_read(ucc_lib_info_t *lib, const char *filename,
                                     ucc_context_config_t **config_p)
{
    ucc_cl_context_config_t *cl_config = NULL;
    int                      i;
    ucc_status_t             status;
    ucc_context_config_t    *config;

    if (filename != NULL) {
        ucc_error("read from file is not implemented");
        return UCC_ERR_NOT_IMPLEMENTED;
    }
    config = (ucc_context_config_t *)ucc_malloc(sizeof(ucc_context_config_t),
                                                "ctx_config");
    if (config == NULL) {
        ucc_error("failed to allocate %zd bytes for context config",
                  sizeof(ucc_context_config_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_config;
    }
    config->lib     = lib;
    config->configs = (ucc_cl_context_config_t **)ucc_calloc(
        lib->n_cl_libs_opened, sizeof(ucc_cl_context_config_t *),
        "cl_configs_array");
    if (config->configs == NULL) {
        ucc_error("failed to allocate %zd bytes for cl configs array",
                  sizeof(ucc_cl_context_config_t *));
        status = UCC_ERR_NO_MEMORY;
        goto err_configs;
    }

    config->n_cl_cfg = 0;
    for (i = 0; i < lib->n_cl_libs_opened; i++) {
        ucc_assert(NULL != lib->cl_libs[i]->iface->cl_context_config.table);
        status =
            ucc_cl_context_config_read(lib->cl_libs[i], config, &cl_config);
        if (UCC_OK != status) {
            ucc_error("failed to read CL \"%s\" context configuration",
                      lib->cl_libs[i]->iface->super.name);
            goto err_config_i;
        }
        config->configs[config->n_cl_cfg]         = cl_config;
        config->n_cl_cfg++;
    }
    *config_p   = config;
    return UCC_OK;

err_config_i:
    for (i = i - 1; i >= 0; i--) {
        ucc_base_config_release(&config->configs[i]->super);
    }
err_configs:
    ucc_free(config->configs);

err_config:
    ucc_free(config);
    return status;
}

/* Look up the cl_context_config in the array of configs based on the
   cl_type. returns NULL if not found */
static inline ucc_cl_context_config_t *
find_cl_context_config(ucc_context_config_t *cfg, ucc_cl_type_t cl_type)
{
    int i;
    for (i = 0; i < cfg->n_cl_cfg; i++) {
        if (cfg->configs[i] &&
            (cl_type == cfg->configs[i]->cl_lib->iface->type)) {
            return cfg->configs[i];
        }
    }
    return NULL;
}

/* Modifies the ucc_context configuration.
   If user sets cls="all" then this means that the parameter  "name" should
   be modified in ALL available CLS. In this case we loop over all of them,
   and if error is reported by any CL we bail and report error to the user.

   If user passes a comma separated list of CLs, then we go over the list
   and apply modifications to the specified CLs only. */
ucc_status_t ucc_context_config_modify(ucc_context_config_t *config,
                                       const char *cls, const char *name,
                                       const char *value)
{
    int                      i;
    ucc_status_t             status;
    ucc_cl_context_config_t *cl_cfg;
    if (0 != strcmp(cls, "all")) {
        ucc_cl_type_t *required_cls;
        int            n_required_cls;
        status = ucc_parse_cls_string(cls, &required_cls, &n_required_cls);
        if (UCC_OK != status) {
            ucc_error("failed to parse cls string: %s", cls);
            return status;
        }
        for (i = 0; i < n_required_cls; i++) {
            cl_cfg = find_cl_context_config(config, required_cls[i]);
            if (!cl_cfg) {
                ucc_error("required CL %s is not part of the context",
                          ucc_cl_names[required_cls[i]]);
                return UCC_ERR_INVALID_PARAM;
            }
            status = ucc_config_parser_set_value(
                cl_cfg, cl_cfg->cl_lib->iface->cl_context_config.table, name,
                value);
            if (UCC_OK != status) {
                ucc_error("failed to modify CL \"%s\" configuration, name %s, "
                          "value %s",
                          cl_cfg->cl_lib->iface->super.name, name, value);
                return status;
            }
        }
        ucc_free(required_cls);
    } else {
        for (i = 0; i < config->n_cl_cfg; i++) {
            if (config->configs[i]) {
                status = ucc_config_parser_set_value(
                    config->configs[i],
                    config->lib->cl_libs[i]->iface->cl_context_config.table, name,
                    value);
                if (UCC_OK != status) {
                    ucc_error("failed to modify CL \"%s\" configuration, name "
                              "%s, value %s",
                              config->lib->cl_libs[i]->iface->super.name, name,
                              value);
                    return status;
                }
            }
        }
    }
    return UCC_OK;
}

void ucc_context_config_release(ucc_context_config_t *config)
{
    int i;
    for (i = 0; i < config->n_cl_cfg; i++) {
        if (!config->configs[i]) {
            continue;
        }
        ucc_base_config_release(&config->configs[i]->super);
    }
    ucc_free(config->configs);
    ucc_free(config);
}

/* The function prints the configuration of UCC context.
   The ucc_context is a combination of contexts of different
   (potentially multiple) CLs.

   If HEADER flag is required - print it once passing user "title"
   variable. For other cl_contexts use CL name as title so that
   the printed output was clear for a user */
void ucc_context_config_print(const ucc_context_config_h config, FILE *stream,
                              const char *title,
                              ucc_config_print_flags_t print_flags)
{
    int i;
    int print_header = print_flags & UCC_CONFIG_PRINT_HEADER;
    int flags        = print_flags;
    /* cl_context_configs will always be printed with HEADER using
       CL name as title */
    flags |= UCC_CONFIG_PRINT_HEADER;

    for (i = 0; i < config->n_cl_cfg; i++) {
        if (!config->configs[i]) {
            continue;
        }
        if (print_header) {
            print_header = 0;
            ucc_config_parser_print_opts(
                stream, title, config->configs[i],
                config->lib->cl_libs[i]->iface->cl_context_config.table,
                config->lib->cl_libs[i]->iface->cl_context_config.prefix,
                config->lib->full_prefix, UCC_CONFIG_PRINT_HEADER);
        }

        ucc_config_parser_print_opts(
            stream, config->lib->cl_libs[i]->iface->cl_context_config.name,
            config->configs[i],
            config->lib->cl_libs[i]->iface->cl_context_config.table,
            config->lib->cl_libs[i]->iface->cl_context_config.prefix,
            config->lib->full_prefix, (ucc_config_print_flags_t)flags);
    }
}

static inline void ucc_copy_context_params(ucc_context_params_t *dst,
                                           const ucc_context_params_t *src)
{
    dst->mask = src->mask;
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_TYPE, ctx_type);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_COLL_OOB, oob);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_ID, ctx_id);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_COLL_SYNC_TYPE,
                            sync_type);
}

static ucc_status_t ucc_create_tl_contexts(ucc_context_t *ctx,
                                           ucc_context_config_t *ctx_config,
                                           ucc_base_context_params_t b_params)
{
    ucc_lib_info_t *lib = ctx->lib;
    ucc_tl_lib_t *tl_lib;
    ucc_tl_context_config_t *tl_config;
    ucc_base_context_t       *b_ctx;
    int i, num_tls;
    ucc_status_t status;

    num_tls = lib->n_tl_libs_opened;
    ctx->tl_ctx = (ucc_tl_context_t **)ucc_malloc(
        sizeof(ucc_tl_context_t *) * num_tls, "tl_ctx_array");
    if (!ctx->tl_ctx) {
        ucc_error("failed to allocate %zd bytes for tl_ctx array",
                  sizeof(ucc_tl_context_t *) * num_tls);
        return UCC_ERR_NO_MEMORY;
    }
    ctx->n_tl_ctx = 0;
    for (i = 0; i < lib->n_tl_libs_opened; i++) {
        tl_lib = lib->tl_libs[i];
        ucc_assert(NULL != tl_lib->iface->tl_context_config.table);
        status =
            ucc_tl_context_config_read(tl_lib, ctx_config, &tl_config);
        if (UCC_OK != status) {
            ucc_warn("failed to read TL \"%s\" context configuration",
                     tl_lib->iface->super.name);
            continue;
        }
        status = tl_lib->iface->context.create(
            &b_params, &tl_config->super, &b_ctx);
        ucc_base_config_release(&tl_config->super);
        if (UCC_OK != status) {
            ucc_warn("failed to create tl context for %s",
                      tl_lib->iface->super.name);
            continue;
        }
        ctx->tl_ctx[ctx->n_tl_ctx] = ucc_derived_of(b_ctx, ucc_tl_context_t);
        ctx->n_tl_ctx++;
    }
    return UCC_OK;
}

ucc_status_t ucc_context_create(ucc_lib_h lib,
                                const ucc_context_params_t *params,
                                const ucc_context_config_h  config,
                                ucc_context_h *context)
{
    ucc_base_context_params_t b_params;
    ucc_base_context_t       *b_ctx;
    ucc_cl_lib_t             *cl_lib;
    ucc_context_t            *ctx;
    ucc_status_t              status;
    uint64_t                  i;
    int                       num_cls;

    num_cls = config->n_cl_cfg;
    ctx     = ucc_malloc(sizeof(ucc_context_t), "ucc_context");
    if (!ctx) {
        ucc_error("failed to allocate %zd bytes for ucc_context",
                  sizeof(ucc_context_t));
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    ctx->lib                     = lib;
    ctx->progress_array          = NULL;
    ctx->progress_array_size     = 0;
    ctx->progress_array_max_size = 0;
    ucc_copy_context_params(&ctx->params, params);
    ucc_copy_context_params(&b_params.params, params);
    b_params.context           = ctx;
    b_params.estimated_num_eps = 0; //TODO
    b_params.estimated_num_ppn = 0; //TODO
    b_params.prefix            = lib->full_prefix;
    b_params.thread_mode       = lib->attr.thread_mode;
    status = ucc_create_tl_contexts(ctx, config, b_params);
    if (UCC_OK != status) {
        /* only critical error could have happened - bail */
        ucc_error("failed to create tl contexts");
        goto error_ctx;
    }

    ctx->cl_ctx = (ucc_cl_context_t **)ucc_malloc(
        sizeof(ucc_cl_context_t *) * num_cls, "cl_ctx_array");
    if (!ctx->cl_ctx) {
        ucc_error("failed to allocate %zd bytes for cl_ctx array",
                  sizeof(ucc_cl_context_t *) * num_cls);
        status = UCC_ERR_NO_MEMORY;
        goto error_ctx;
    }
    ctx->n_cl_ctx = 0;
    for (i = 0; i < num_cls; i++) {
        cl_lib = config->configs[i]->cl_lib;
        status = cl_lib->iface->context.create(
            &b_params, &config->configs[i]->super, &b_ctx);
        if (UCC_OK != status) {
            if (lib->specific_cls_requested) {
                ucc_error("failed to create cl context for %s",
                          cl_lib->iface->super.name);
                goto error_ctx_create;
            } else {
                ucc_warn("failed to create cl context for %s, skipping",
                         cl_lib->iface->super.name);
                continue;
            }
        }
        ctx->cl_ctx[ctx->n_cl_ctx] = ucc_derived_of(b_ctx, ucc_cl_context_t);
        ctx->n_cl_ctx++;
    }
    if (0 == ctx->n_cl_ctx) {
        ucc_error("no CL context created in ucc_context_create");
        status = UCC_ERR_NO_MESSAGE;
        goto error_ctx;
    }

    /* Initialize ctx thread mode:
       if context is EXCLUSIVE then thread_mode is always SINGLE,
       otherwise it is  inherited from lib */
    ctx->thread_mode = ((params->ctx_type == UCC_CONTEXT_EXCLUSIVE) &&
                        (params->mask & UCC_CONTEXT_PARAM_FIELD_TYPE))
                           ? UCC_THREAD_SINGLE
                           : lib->attr.thread_mode;
    status           = ucc_progress_queue_init(&ctx->pq, ctx->thread_mode);
    if (UCC_OK != status) {
        ucc_error("failed to init progress queue for context %p", ctx);
        goto error_ctx_create;
    }
    ucc_info("created ucc context %p for lib %s", ctx, lib->full_prefix);
    *context = ctx;
    return UCC_OK;

error_ctx_create:
    for (i = i - 1; i >= 0; i--) {
        config->configs[i]->cl_lib->iface->context.destroy(
            &ctx->cl_ctx[i]->super);
    }
    ucc_free(ctx->cl_ctx);
error_ctx:
    ucc_free(ctx);
error:
    return status;
}

ucc_status_t ucc_context_destroy(ucc_context_t *context)
{
    ucc_cl_context_t *cl_ctx;
    ucc_cl_lib_t     *cl_lib;
    ucc_tl_context_t *tl_ctx;
    ucc_tl_lib_t     *tl_lib;
    int               i;
    for (i = 0; i < context->n_cl_ctx; i++) {
        cl_ctx = context->cl_ctx[i];
        cl_lib = ucc_derived_of(cl_ctx->super.lib, ucc_cl_lib_t);
        cl_lib->iface->context.destroy(&cl_ctx->super);
    }
    ucc_free(context->cl_ctx);

    for (i = 0; i < context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        tl_lib = ucc_derived_of(tl_ctx->super.lib, ucc_tl_lib_t);
        if (tl_ctx->ref_count != 0 ) {
            ucc_warn("tl ctx %s is still in use", tl_lib->iface->super.name);
        }
        tl_lib->iface->context.destroy(&tl_ctx->super);
    }
    ucc_progress_queue_finalize(context->pq);
    ucc_free(context->tl_ctx);
    ucc_free(context);
    return UCC_OK;
}

ucc_status_t ucc_context_progress_register(ucc_context_t *ctx,
                                           ucc_context_progress_fn_t fn,
                                           void *progress_arg)
{
    int next_pos = ctx->progress_array_size;
    if (next_pos == ctx->progress_array_max_size) {
        ctx->progress_array_max_size += 8;
        ctx->progress_array = ucc_realloc(ctx->progress_array,
                                          ctx->progress_array_max_size *
                                              sizeof(ucc_context_progress_t),
                                          "progress_array");
        if (!ctx->progress_array) {
            ucc_error("failed to allocate %zd bytes for progress array",
                      ctx->progress_array_max_size *
                          sizeof(ucc_context_progress_t));
            return UCC_ERR_NO_MEMORY;
        }
    }
    ctx->progress_array[next_pos].progress_fn  = fn;
    ctx->progress_array[next_pos].progress_arg = progress_arg;
    ctx->progress_array_size++;
    return UCC_OK;
}

void ucc_context_progress_deregister(ucc_context_t *ctx,
                                     ucc_context_progress_fn_t fn,
                                     void *progress_arg)
{
    int i, j;
    for (i = 0; i < ctx->progress_array_size; i++) {
        if (ctx->progress_array[i].progress_fn == fn &&
            ctx->progress_array[i].progress_arg == progress_arg) {
            for (j = i; j < ctx->progress_array_size - 1; j++) {
                ctx->progress_array[j] = ctx->progress_array[j + 1];
            }
            ctx->progress_array_size--;
            return;
        }
    }
    ucc_assert(0);
}

ucc_status_t ucc_context_progress(ucc_context_h context)
{
    int                     pa_size = context->progress_array_size;
    ucc_context_progress_t *p;
    ucc_status_t            status;
    int                     i;
    /* progress registered progress fns */
    for (i = 0; i < pa_size; i++) {
        p = &context->progress_array[i];
        /* registred progress fn returns 0 if no communication is made,
           or positive value otherwise */
        p->progress_fn(p->progress_arg);
    }

    /* the fn below returns int - number of completed tasks.
       TODO : do we need to handle it ? Maybe return to user
       as int as well? */
    status = ucc_progress_queue(context->pq);
    return (status >= 0 ? UCC_OK : status);
}
