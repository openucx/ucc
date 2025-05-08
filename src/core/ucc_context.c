/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_context.h"
#include "components/cl/ucc_cl.h"
#include "components/tl/ucc_tl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_list.h"
#include "utils/ucc_string.h"
#include "ucc_progress_queue.h"

static uint32_t ucc_context_seq_num = 0;
static ucc_config_field_t ucc_context_config_table[] = {
    {"ESTIMATED_NUM_EPS", "0",
     "An optimization hint of how many endpoints will be created on this "
     "context",
     ucc_offsetof(ucc_context_config_t, estimated_num_eps),
     UCC_CONFIG_TYPE_UINT},

    {"LOCK_FREE_PROGRESS_Q", "0",
     "Enable lock free progress queue optimization",
     ucc_offsetof(ucc_context_config_t, lock_free_progress_q),
     UCC_CONFIG_TYPE_UINT},

    {"ESTIMATED_NUM_PPN", "0",
     "An optimization hint of how many endpoints created on this context reside"
     " on the same node",
     ucc_offsetof(ucc_context_config_t, estimated_num_ppn),
     UCC_CONFIG_TYPE_UINT},

    {"TEAM_IDS_POOL_SIZE", "32",
     "Defines the size of the team_id_pool. The number of coexisting unique "
     "team ids for a single process is team_ids_pool_size*64. This parameter "
     "is relevant when internal team id allocation takes place.",
     ucc_offsetof(ucc_context_config_t, team_ids_pool_size),
     UCC_CONFIG_TYPE_UINT},

    {"INTERNAL_OOB", "1",
     "Use internal OOB transport for team creation. Available for ucc_context "
     "is configured with OOB (global mode). 0 - disable, 1 - try, 2 - force.",
     ucc_offsetof(ucc_context_config_t, internal_oob), UCC_CONFIG_TYPE_UINT},

    {"THROTTLE_PROGRESS", "1000",
     "Throttle UCC progress to every <n>th invocation",
     ucc_offsetof(ucc_context_config_t, throttle_progress),
     UCC_CONFIG_TYPE_UINT},

    {NULL}};
UCC_CONFIG_REGISTER_TABLE(ucc_context_config_table, "UCC context", NULL,
                          ucc_context_config_t, &ucc_config_global_list);

ucc_status_t ucc_context_config_read(ucc_lib_info_t *lib, const char *filename,
                                     ucc_context_config_t **config_p)
{
    ucc_cl_context_config_t *cl_config = NULL;
    ucc_tl_context_config_t *tl_config = NULL;
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
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_config_parser_fill_opts(config,
                                         UCC_CONFIG_GET_TABLE(ucc_context_config_table),
                                         lib->full_prefix, 0);
    if (status != UCC_OK) {
        ucc_error("failed to read UCC core context config");
        goto err_config;
    }

    config->lib     = lib;
    config->cl_cfgs = (ucc_cl_context_config_t **)ucc_calloc(
        lib->n_cl_libs_opened, sizeof(ucc_cl_context_config_t *),
        "cl_configs_array");
    if (config->cl_cfgs == NULL) {
        ucc_error("failed to allocate %zd bytes for cl configs array",
                  sizeof(ucc_cl_context_config_t *));
        status = UCC_ERR_NO_MEMORY;
        goto err_config;
    }

    config->tl_cfgs = (ucc_tl_context_config_t **)ucc_calloc(
        lib->n_tl_libs_opened, sizeof(ucc_tl_context_config_t *),
        "tl_configs_array");
    if (config->tl_cfgs == NULL) {
        ucc_error("failed to allocate %zd bytes for tl configs array",
                  sizeof(ucc_tl_context_config_t *));
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
            goto err_cl_config;
        }
        config->cl_cfgs[config->n_cl_cfg] = cl_config;
        config->n_cl_cfg++;
    }

    config->n_tl_cfg = 0;
    for (i = 0; i < lib->n_tl_libs_opened; i++) {
        ucc_assert(NULL != lib->tl_libs[i]->iface->tl_context_config.table);
        status =
            ucc_tl_context_config_read(lib->tl_libs[i], config, &tl_config);
        if (UCC_OK != status) {
            ucc_error("failed to read TL \"%s\" context configuration",
                      lib->tl_libs[i]->iface->super.name);
            goto err_tl_config;
        }
        config->tl_cfgs[config->n_tl_cfg] = tl_config;
        config->n_tl_cfg++;
    }

    *config_p   = config;
    return UCC_OK;

err_tl_config:
    for (i = 0; i < config->n_tl_cfg; i++) {
        ucc_base_config_release(&config->tl_cfgs[i]->super.super);
    }

err_cl_config:
    for (i = 0; i < config->n_cl_cfg; i++) {
        ucc_base_config_release(&config->cl_cfgs[i]->super.super);
    }
    ucc_free(config->tl_cfgs);

err_configs:
    ucc_free(config->cl_cfgs);

err_config:
    ucc_free(config);
    return status;
}

static inline ucc_cl_context_config_t *
find_cl_context_config(ucc_context_config_t *cfg, const char *name)
{
    int i;
    for (i = 0; i < cfg->n_cl_cfg; i++) {
        if (cfg->cl_cfgs[i] &&
            (0 == strcmp(cfg->cl_cfgs[i]->cl_lib->iface->super.name,
                         name))) {
            return cfg->cl_cfgs[i];
        }
    }
    return NULL;
}

static inline ucc_tl_context_config_t *
find_tl_context_config(ucc_context_config_t *cfg, const char *name)
{
    int i;
    for (i = 0; i < cfg->n_tl_cfg; i++) {
        if (cfg->tl_cfgs[i] &&
            (0 == strcmp(cfg->tl_cfgs[i]->tl_lib->iface->super.name,
                         name))) {
            return cfg->tl_cfgs[i];
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
                                       const char *components,
                                       const char *name,
                                       const char *value)
{
    int                      i;
    ucc_status_t             status;
    ucc_cl_context_config_t *cl_cfg;
    ucc_tl_context_config_t *tl_cfg;
    char                   **tokens;
    const char              *component;
    unsigned                 n_tokens;

    if (NULL == components) {
        /* cls is NULL means modify core ucc context config */
        status = ucc_config_parser_set_value(config, ucc_context_config_table, name,
                                             value);
        if (UCC_OK != status) {
            ucc_error("failed to modify CORE configuration, name %s, value %s",
                      name, value);;
            return status;
        }
    } else if (0 != strcmp(components, "all")) {
        tokens = ucc_str_split(components, ",");
        if (!tokens) {
            return UCC_ERR_INVALID_PARAM;
        }
        n_tokens = ucc_str_split_count(tokens);

        for (i = 0; i < n_tokens; i++) {
            if (strlen(tokens[i]) < 4) {
                status = UCC_ERR_INVALID_PARAM;
                goto err_input;
            }
            component = tokens[i] + 3;
            if (0 == strncmp(tokens[i], "cl/", 3)) {
                cl_cfg = find_cl_context_config(config, component);
                if (!cl_cfg) {
                    ucc_info("required CL %s is not part of the context",
                             component);
                    status = UCC_ERR_NOT_FOUND;
                    goto err;
                }
                status = ucc_config_parser_set_value(
                    cl_cfg, cl_cfg->cl_lib->iface->cl_context_config.table, name,
                    value);
                if (UCC_OK != status) {
                    ucc_error("failed to modify CL \"%s\" configuration, name %s, "
                              "value %s", component, name, value);
                    goto err;
                }
            } else if (0 == strncmp(tokens[i], "tl/", 3)) {
                tl_cfg = find_tl_context_config(config, component);
                if (!tl_cfg) {
                    ucc_info("required TL %s is not part of the context",
                             component);
                    status = UCC_ERR_NOT_FOUND;
                    goto err;
                }
                status = ucc_config_parser_set_value(
                    tl_cfg, tl_cfg->tl_lib->iface->tl_context_config.table, name,
                    value);
                if (UCC_OK != status) {
                    ucc_error("failed to modify TL \"%s\" configuration, name %s, "
                              "value %s", component, name, value);
                    goto err;
                }
            } else {
                status = UCC_ERR_INVALID_PARAM;
                ucc_error("invalid component name %s", tokens[i]);
                goto err_input;
            }
        }
        ucc_str_split_free(tokens);
    } else {
        for (i = 0; i < config->n_cl_cfg; i++) {
            if (config->cl_cfgs[i]) {
                status = ucc_config_parser_set_value(
                    config->cl_cfgs[i],
                    config->cl_cfgs[i]->cl_lib->iface->cl_context_config.table,
                    name, value);
                if (UCC_OK != status) {
                    ucc_error("failed to modify CL \"%s\" configuration, name "
                              "%s, value %s",
                              config->cl_cfgs[i]->cl_lib->iface->super.name, name,
                              value);
                    return status;
                }
            }
        }
    }
    return UCC_OK;

err_input:
    ucc_error("incorrect component name %s is provided. "
              "expected cl/<cl_name> or tl/<tl_name>.",
              tokens[i]);
err:
    ucc_str_split_free(tokens);
    return status;
}

void ucc_context_config_release(ucc_context_config_t *config)
{
    int i;

    for (i = 0; i < config->n_tl_cfg; i++) {
        ucc_base_config_release(&config->tl_cfgs[i]->super.super);
    }

    for (i = 0; i < config->n_cl_cfg; i++) {
        ucc_base_config_release(&config->cl_cfgs[i]->super.super);
    }
    ucc_free(config->tl_cfgs);
    ucc_free(config->cl_cfgs);
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

    if (print_header) {
        ucc_config_parser_print_opts(
            stream, title, config,
            ucc_context_config_table, "",
            config->lib->full_prefix, UCC_CONFIG_PRINT_HEADER);
    }

    ucc_config_parser_print_opts(
        stream, "CORE",
        config, ucc_context_config_table, "",
        config->lib->full_prefix, (ucc_config_print_flags_t)flags);

    for (i = 0; i < config->n_cl_cfg; i++) {
        ucc_config_parser_print_opts(
            stream, config->lib->cl_libs[i]->iface->cl_context_config.name,
            config->cl_cfgs[i],
            config->lib->cl_libs[i]->iface->cl_context_config.table,
            config->lib->cl_libs[i]->iface->cl_context_config.prefix,
            config->lib->full_prefix, (ucc_config_print_flags_t)flags);
    }
}

static inline void ucc_copy_context_params(ucc_context_params_t *dst,
                                           const ucc_context_params_t *src)
{
    dst->mask = src->mask;
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_TYPE, type);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_OOB, oob);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_ID, ctx_id);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_SYNC_TYPE,
                            sync_type);
    UCC_COPY_PARAM_BY_FIELD(dst, src, UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS, mem_params);
}

static ucc_status_t ucc_create_tl_contexts(ucc_context_t *ctx,
                                           ucc_context_config_t *ctx_config,
                                           ucc_base_context_params_t b_params)
{
    ucc_lib_info_t     *lib = ctx->lib;
    ucc_tl_lib_t       *tl_lib;
    ucc_base_context_t *b_ctx;
    int                 i, num_tls;
    ucc_status_t        status;
    ucc_base_lib_attr_t attr;
    int                 ctx_service_team;

    ctx_service_team = ctx_config->internal_oob &&
        b_params.params.mask & UCC_CONTEXT_PARAM_FIELD_OOB;
    num_tls = ctx_config->n_tl_cfg;
    ctx->tl_ctx = (ucc_tl_context_t **)ucc_malloc(
        sizeof(ucc_tl_context_t *) * num_tls, "tl_ctx_array");
    if (!ctx->tl_ctx) {
        ucc_error("failed to allocate %zd bytes for tl_ctx array",
                  sizeof(ucc_tl_context_t *) * num_tls);
        return UCC_ERR_NO_MEMORY;
    }
    ctx->n_tl_ctx = 0;
    for (i = 0; i < num_tls; i++) {
        tl_lib = ctx_config->tl_cfgs[i]->tl_lib;
        status = tl_lib->iface->lib.get_attr(&tl_lib->super, &attr);
        if (UCC_OK != status) {
            ucc_error("failed to query tl lib %s attr",
                       tl_lib->iface->super.name);
            return status;
        }
        if ((attr.flags & UCC_BASE_LIB_FLAG_CTX_SERVICE_TEAM_REQUIRED) &&
            (!ctx_service_team)) {
            ucc_debug("can not create tl/%s context because context service "
                      "team is not available for it",
                      tl_lib->iface->super.name);
            continue;
        }
        // coverity[overrun-buffer-val:FALSE]
        status = tl_lib->iface->context.create(
            &b_params, &ctx_config->tl_cfgs[i]->super.super, &b_ctx);
        if (UCC_OK != status) {
            /* UCC_ERR_LAST means component was disabled via TUNE param:
               don't print warning. */
            if (UCC_ERR_LAST != status) {
                if (ucc_tl_is_required(lib, tl_lib->iface, 1)) {
                    ucc_error("failed to create tl context for %s",
                              tl_lib->iface->super.name);
                } else {
                    ucc_debug("failed to create tl context for %s",
                              tl_lib->iface->super.name);
                }
            }
            continue;
        }
        ctx->tl_ctx[ctx->n_tl_ctx] = ucc_derived_of(b_ctx, ucc_tl_context_t);
        ctx->n_tl_ctx++;
    }
    if (ctx->n_tl_ctx == 0) {
        ucc_error("no tl contexts were created");
        status = UCC_ERR_NOT_FOUND;
        goto err;
    }
    /* build the list of names of all available tl contexts.
       This is a convenience data struct for CLs */
    ctx->all_tls.count = ctx->n_tl_ctx;
    ctx->all_tls.names = ucc_malloc(sizeof(char*) * ctx->n_tl_ctx, "all_tls");
    if (!ctx->all_tls.names) {
        ucc_error("failed to allocate %zd bytes for all_tls names",
                  sizeof(char*) * ctx->n_tl_ctx);
        status = UCC_ERR_NO_MEMORY;
        goto err;

    }
    for (i = 0; i < ctx->n_tl_ctx; i++) {
        ctx->all_tls.names[i] = (char*)ucc_derived_of(ctx->tl_ctx[i]->super.lib,
                                               ucc_tl_lib_t)->iface->super.name;
    }
    return UCC_OK;
err:
    for (i = 0; i < ctx->n_tl_ctx; i++) {
        tl_lib = ucc_derived_of(ctx->tl_ctx[i]->super.lib, ucc_tl_lib_t);
        tl_lib->iface->context.destroy(&ctx->tl_ctx[i]->super);
    }
    ucc_free(ctx->tl_ctx);
    return status;
}

ucc_status_t ucc_core_addr_exchange(ucc_context_t *context, ucc_oob_coll_t *oob,
                                    ucc_addr_storage_t *addr_storage)
{
    size_t             *addr_lens;
    ucc_context_attr_t  attr;
    ucc_status_t        status;
    ucc_rank_t          i;
    size_t              max_addrlen;

poll:
    if (addr_storage->oob_req) {
        status = oob->req_test(addr_storage->oob_req);
        if (status < 0) {
            oob->req_free(addr_storage->oob_req);
            ucc_error("oob req test failed during team addr exchange");
            return status;
        } else if (UCC_INPROGRESS == status) {
            return status;
        }
        oob->req_free(addr_storage->oob_req);
        addr_storage->oob_req = NULL;
    }
    if (0 == addr_storage->addr_len) {
        if (NULL == addr_storage->storage) {
            addr_storage->size = oob->n_oob_eps;
            attr.mask          = UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN |
                                 UCC_CONTEXT_ATTR_FIELD_CTX_ADDR;
            status = ucc_context_get_attr(context, &attr);
            if (UCC_OK != status) {
                ucc_error("failed to query ctx address");
                return status;
            }
            addr_storage->storage = ucc_malloc(
                addr_storage->size * sizeof(size_t), "max_addrlen_tmp");
            if (!addr_storage->storage) {
                ucc_error(
                    "failed to allocate %zd bytes for max_addrlen tmp storage",
                    addr_storage->size * sizeof(size_t));
                return UCC_ERR_NO_MEMORY;
            }

            status = oob->allgather(&context->attr.ctx_addr_len,
                                    addr_storage->storage, sizeof(size_t),
                                    oob->coll_info, &addr_storage->oob_req);
            if (UCC_OK != status) {
                ucc_error("failed to start oob allgather");
                return status;
            }
            goto poll;
        }
        addr_lens = (size_t *)addr_storage->storage;
        ucc_assert(addr_storage->storage != NULL);
        for (i = 0; i < addr_storage->size; i++) {
            if (addr_lens[i] > addr_storage->addr_len) {
                addr_storage->addr_len = addr_lens[i];
            }
        }
        if (addr_storage->addr_len == 0 ) {
            ucc_free(addr_storage->storage);
            addr_storage->storage = NULL;
            return UCC_OK;
        }
        max_addrlen = addr_storage->addr_len;
        addr_storage->storage =
            ucc_realloc(addr_storage->storage,
                        (addr_storage->size + 1) * max_addrlen, "addr_storage");
        if (!addr_storage->storage) {
            ucc_error("failed to allocate %zd bytes for addr storage",
                      addr_storage->size * max_addrlen);
            return UCC_ERR_NO_MEMORY;
        }
        memcpy(
            PTR_OFFSET(addr_storage->storage, max_addrlen * addr_storage->size),
            context->attr.ctx_addr, context->attr.ctx_addr_len);
        status = oob->allgather(
            PTR_OFFSET(addr_storage->storage, max_addrlen * addr_storage->size),
            addr_storage->storage, max_addrlen, oob->coll_info,
            &addr_storage->oob_req);
        if (UCC_OK != status) {
            ucc_error("failed to start oob allgather");
            return status;
        }
        goto poll;
    }
    ucc_assert(addr_storage->addr_len);

    {
        /* Compute storage rank and check proc info uniqeness */
        ucc_rank_t r = UCC_RANK_MAX;
        int j;
        ucc_context_addr_header_t *h, *h0;

        addr_storage->flags = UCC_ADDR_STORAGE_FLAG_TLS_SYMMETRIC;
        h0 = UCC_ADDR_STORAGE_RANK_HEADER(addr_storage, 0);
        for (i = 0; i < addr_storage->size; i++) {
            h = UCC_ADDR_STORAGE_RANK_HEADER(addr_storage, i);
            if (UCC_CTX_ID_EQUAL(context->id, h->ctx_id)) {
                /* We should find local id only once. However, due to node hashing
                   there is tiny chance of collision: check it */
                if (r != UCC_RANK_MAX) {
                    ucc_error("proc info collision: %d %d", r, i);
                    return UCC_ERR_NO_MESSAGE;
                }
                r = i;
            }
            if (h->n_components == h0->n_components) {
                /*check if TLs array is the same*/
                for (j = 0; j < h->n_components; j++) {
                    if (h->components[j].id != h0->components[j].id) {
                        addr_storage->flags = 0;
                        break;
                    }
                }
            } else {
                addr_storage->flags = 0;
            }
        }
        addr_storage->rank = r;
    }

    return UCC_OK;
}

static void remove_tl_ctx_from_array(ucc_tl_context_t **array, unsigned *size,
                                     ucc_tl_context_t *tl_ctx)
{
    int i;

    for (i = 0; i < (*size); i++) {
        if (array[i] == tl_ctx) {
            break;
        }
    }
    if (i == (*size)) {
        /* given tl_ctx is not part of array */
        return;
    }
    /* decrement array size and do cyclic shift */
    (*size)--;
    for (; i < (*size); i++) {
        array[i] = array[i + 1];
    }
}

ucc_status_t ucc_context_create_proc_info(ucc_lib_h                   lib,
                                          const ucc_context_params_t *params,
                                          const ucc_context_config_h  config,
                                          ucc_context_h              *context,
                                          ucc_proc_info_t            *proc_info)
{
    uint32_t                   topo_required       = 0;
    uint64_t                   created_ctx_counter = 0;
    ucc_base_context_params_t  b_params;
    ucc_base_context_t        *b_ctx;
    ucc_base_ctx_attr_t        c_attr;
    ucc_cl_lib_attr_t          l_attr;
    ucc_cl_lib_t              *cl_lib;
    ucc_tl_context_t          *tl_ctx;
    ucc_tl_lib_t              *tl_lib;
    ucc_context_t             *ctx;
    ucc_status_t               status;
    uint64_t                   i, j, n_tl_ctx;
    int                        num_cls;

    num_cls = config->n_cl_cfg;
    ctx     = ucc_calloc(1, sizeof(ucc_context_t), "ucc_context");
    if (!ctx) {
        ucc_error("failed to allocate %zd bytes for ucc_context",
                  sizeof(ucc_context_t));
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    ctx->throttle_progress = config->throttle_progress;
    ctx->rank              = UCC_RANK_MAX;
    ctx->lib               = lib;
    ctx->ids.pool_size     = config->team_ids_pool_size;
    ucc_list_head_init(&ctx->progress_list);
    ucc_copy_context_params(&ctx->params, params);
    ucc_copy_context_params(&b_params.params, params);
    b_params.context           = ctx;
    b_params.estimated_num_eps = config->estimated_num_eps;
    b_params.estimated_num_ppn = config->estimated_num_ppn;
    b_params.prefix            = lib->full_prefix;
    b_params.thread_mode       = lib->attr.thread_mode;
    if (params->mask & UCC_CONTEXT_PARAM_FIELD_OOB) {
        ctx->rank = params->oob.oob_ep;
    }
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
    ctx->cl_flags = 0;
    for (i = 0; i < num_cls; i++) {
        cl_lib = config->cl_cfgs[i]->cl_lib;
        // coverity[overrun-buffer-val:FALSE]
        status = cl_lib->iface->context.create(
            &b_params, &config->cl_cfgs[i]->super.super, &b_ctx);
        if (UCC_OK != status) {
            if (lib->specific_cls_requested) {
                ucc_error("failed to create cl context for %s",
                          cl_lib->iface->super.name);
                goto error_ctx_create;
            } else {
                /* UCC_ERR_LAST means component was disabled via TUNE param:
                   don't print warning. */
                if (UCC_ERR_LAST != status) {

                    ucc_warn("failed to create cl context for %s, skipping",
                             cl_lib->iface->super.name);
                }
                continue;
            }
        }
        ctx->cl_ctx[ctx->n_cl_ctx] = ucc_derived_of(b_ctx, ucc_cl_context_t);
        ctx->n_cl_ctx++;

        memset(&c_attr, 0, sizeof(c_attr));
        status = cl_lib->iface->context.get_attr(b_ctx, &c_attr);
        if (status != UCC_OK) {
            ucc_error("failed to query context attributes for %s",
                      cl_lib->iface->super.name);
            goto error_ctx_create;
        }
        if (c_attr.topo_required) {
            topo_required = 1;
        }

        memset(&l_attr, 0, sizeof(l_attr));
        status = cl_lib->iface->lib.get_attr(&cl_lib->super, &l_attr.super);
        if (UCC_OK != status) {
            ucc_error("failed to query lib %s attr", cl_lib->iface->super.name);
            goto error_ctx_create;
        }
        ctx->cl_flags |= l_attr.super.flags;
    }
    if (0 == ctx->n_cl_ctx) {
        ucc_error("no CL context created in ucc_context_create");
        status = UCC_ERR_NO_MESSAGE;
        goto error_ctx;
    }

    /* Initialize ctx thread mode:
       if context is EXCLUSIVE then thread_mode is always SINGLE,
       otherwise it is  inherited from lib */
    ctx->thread_mode = ((params->type == UCC_CONTEXT_EXCLUSIVE) &&
                        (params->mask & UCC_CONTEXT_PARAM_FIELD_TYPE))
                           ? UCC_THREAD_SINGLE
                           : lib->attr.thread_mode;
    status           = ucc_progress_queue_init(&ctx->pq, ctx->thread_mode,
                                               config->lock_free_progress_q);
    if (UCC_OK != status) {
        ucc_error("failed to init progress queue for context %p", ctx);
        goto error_ctx_create;
    }
    ctx->id.pi      = *proc_info;
    ctx->id.seq_num = ucc_atomic_fadd32(&ucc_context_seq_num, 1);
    if (params->mask & UCC_CONTEXT_PARAM_FIELD_OOB &&
        params->oob.n_oob_eps > 1) {
        do {
            /* UCC context create is blocking fn, so we can wait here for the
               completion of addr exchange */
            status = ucc_core_addr_exchange(ctx, &ctx->params.oob,
                                            &ctx->addr_storage);
            if (status < 0) {
                ucc_error("failed to exchange addresses during context creation");
                goto error_ctx_create;
            }
        } while (status == UCC_INPROGRESS);

        if (topo_required) {
            /* At least one available CL context reported it needs topo info */
            status = ucc_context_topo_init(&ctx->addr_storage, &ctx->topo);
            if (UCC_OK != status) {
                ucc_free(ctx->addr_storage.storage);
                ucc_error("failed to init ctx topo");
                goto error_ctx_create;
            }
        }
        ucc_assert(ctx->addr_storage.rank == params->oob.oob_ep);
    }
    if (config->internal_oob) {
        if (params->mask & UCC_CONTEXT_PARAM_FIELD_OOB &&
            params->oob.n_oob_eps > 1) {
            ucc_base_team_params_t t_params;
            ucc_base_team_t *      b_team;
            status = ucc_tl_context_get(ctx, "ucp", &ctx->service_ctx);
            if (UCC_OK != status) {
                if (config->internal_oob == 2) {
                    ucc_error(
                        "TL UCP context is not available, service team can "
                        "not be created but was force requested");
                    goto error_ctx_create;
                }
                ucc_debug("TL UCP context is not available, "
                          "service team can not be created");
            } else {
                memset(&t_params.map, 0, sizeof(ucc_ep_map_t));
                memset(&t_params.params, 0, sizeof(ucc_team_params_t));
                t_params.params.mask = UCC_TEAM_PARAM_FIELD_EP |
                                       UCC_TEAM_PARAM_FIELD_EP_RANGE |
                                       UCC_TEAM_PARAM_FIELD_OOB;
                t_params.params.oob.allgather    = ctx->params.oob.allgather;
                t_params.params.oob.req_test     = ctx->params.oob.req_test;
                t_params.params.oob.req_free     = ctx->params.oob.req_free;
                t_params.params.oob.coll_info    = ctx->params.oob.coll_info;
                t_params.params.oob.n_oob_eps    = ctx->params.oob.n_oob_eps;
                t_params.params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
                t_params.params.ep       = ctx->rank;
                t_params.rank            = ctx->rank;
                t_params.size            = ctx->params.oob.n_oob_eps;
                /* CORE scope id - never overlaps with CL type */
                t_params.scope      = UCC_CL_LAST + 1;
                t_params.scope_id   = 0;
                t_params.id         = 0;
                t_params.team       = NULL;
                t_params.map.type   = UCC_EP_MAP_FULL;
                t_params.map.ep_num = t_params.size;
                status            = UCC_TL_CTX_IFACE(ctx->service_ctx)
                             ->team.create_post(&ctx->service_ctx->super,
                                                &t_params, &b_team);
                if (UCC_OK != status) {
                    ucc_error("ctx service team create post failed");
                    goto error_ctx_create;
                }
                do {
                    status = UCC_TL_CTX_IFACE(ctx->service_ctx)
                                 ->team.create_test(b_team);
                } while (UCC_INPROGRESS == status);
                if (status < 0) {
                    ucc_error("failed to create ctx service team");
                    goto error_ctx_create;
                }
                ctx->service_team = ucc_derived_of(b_team, ucc_tl_team_t);
            }
        } else if (config->internal_oob == 2) {
            ucc_error("UCC_INTERNAL_OOB was force requested for context "
                      "without OOB");
            goto error_ctx_create;
        }
    }

    n_tl_ctx = ctx->n_tl_ctx;
    for (i = 0; i < n_tl_ctx; i++) {
        tl_ctx = ctx->tl_ctx[i];
        tl_lib = ucc_derived_of(tl_ctx->super.lib, ucc_tl_lib_t);
        if (tl_lib->iface->context.create_epilog) {
            status = tl_lib->iface->context.create_epilog(&tl_ctx->super);
            if (UCC_OK == status) {
                created_ctx_counter++;
            } else {
                if (ucc_tl_is_required(lib, tl_lib->iface, 1)) {
                    ucc_error("ctx create epilog for %s failed: %s",
                              tl_lib->iface->super.name, ucc_status_string(status));
                    goto error_ctx_create_epilog;
                } else {
                    ucc_debug("ctx create epilog for %s failed: %s",
                              tl_lib->iface->super.name, ucc_status_string(status));
                    tl_lib->iface->context.destroy(&tl_ctx->super);
                    for (j = 0; j < ctx->n_cl_ctx; j++) {
                        remove_tl_ctx_from_array(ctx->cl_ctx[j]->tl_ctxs,
                                                 &ctx->cl_ctx[j]->n_tl_ctxs,
                                                 tl_ctx);
                    }
                    remove_tl_ctx_from_array(ctx->tl_ctx, &ctx->n_tl_ctx,
                                             tl_ctx);
                }
            }
        } else {
            created_ctx_counter++;
        }
    }
    if (0 == created_ctx_counter) {
        ucc_error("no TL context created");
        status = UCC_ERR_NO_RESOURCE;
        goto error_ctx_create_epilog;
    }

    ucc_debug("created ucc context %p for lib %s", ctx, lib->full_prefix);
    *context = ctx;
    return UCC_OK;

error_ctx_create_epilog:
    for (j = 0; j < created_ctx_counter; j++) {
        tl_ctx = ctx->tl_ctx[j];
        tl_lib = ucc_derived_of(tl_ctx->super.lib, ucc_tl_lib_t);
        tl_lib->iface->context.destroy(&tl_ctx->super);
    }
error_ctx_create:
    for (i = 0; i < ctx->n_cl_ctx; i++) {
        config->cl_cfgs[i]->cl_lib->iface->context.destroy(
            &ctx->cl_ctx[i]->super);
    }
    ucc_free(ctx->cl_ctx);
error_ctx:
    ucc_free(ctx);
error:
    return status;
}

ucc_status_t ucc_context_create(ucc_lib_h lib,
                                const ucc_context_params_t *params,
                                const ucc_context_config_h  config,
                                ucc_context_h *context)
{
    return ucc_context_create_proc_info(lib, params, config, context,
                                        &ucc_local_proc);
}

static ucc_status_t ucc_context_free_attr(ucc_context_attr_t *context_attr)
{
    if (context_attr->mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR) {
        ucc_free(context_attr->ctx_addr);
    }
    return UCC_OK;
}

ucc_status_t ucc_context_destroy(ucc_context_t *context)
{
    ucc_cl_context_t *cl_ctx;
    ucc_cl_lib_t     *cl_lib;
    ucc_tl_context_t *tl_ctx;
    ucc_tl_lib_t     *tl_lib;
    int               i;
    ucc_status_t      status;

    if (UCC_OK != ucc_context_free_attr(&context->attr)) {
        ucc_error("failed to free context attributes");
    }
    for (i = 0; i < context->n_cl_ctx; i++) {
        cl_ctx = context->cl_ctx[i];
        cl_lib = ucc_derived_of(cl_ctx->super.lib, ucc_cl_lib_t);
        cl_lib->iface->context.destroy(&cl_ctx->super);
    }
    ucc_free(context->cl_ctx);

    for (i = 0; i < context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        if ((context->service_team) && (tl_ctx == context->service_ctx)) {
            /* skip service context cause service team might be
               in use by other contexts */
            continue;
        }
        tl_lib = ucc_derived_of(tl_ctx->super.lib, ucc_tl_lib_t);
        if (tl_ctx->ref_count != 0 ) {
            ucc_info("tl ctx %s is still in use", tl_lib->iface->super.name);
        }
        tl_lib->iface->context.destroy(&tl_ctx->super);
    }

    if (context->service_team) {
        /* service team can now be safely destroyed since all other contexts
           have been cleaned up and can no longer use the service team */
        while (UCC_INPROGRESS ==
               (status = UCC_TL_CTX_IFACE(context->service_ctx)
                             ->team.destroy(&context->service_team->super))) {
            ucc_context_progress(context);
        }
        if (status < 0) {
            ucc_error("failed to destroy ctx service team");
        }
        ucc_tl_context_put(context->service_ctx);
        tl_lib = ucc_derived_of(context->service_ctx->super.lib, ucc_tl_lib_t);
        if (context->service_ctx->ref_count != 0 ) {
            ucc_info("tl ctx %s is still in use", tl_lib->iface->super.name);
        }
        tl_lib->iface->context.destroy(&context->service_ctx->super);
    }

    ucc_context_topo_cleanup(context->topo);
    ucc_progress_queue_finalize(context->pq);
    ucc_free(context->addr_storage.storage);
    ucc_free(context->all_tls.names);
    ucc_free(context->tl_ctx);
    ucc_free(context->ids.pool);
    ucc_free(context);
    return UCC_OK;
}

typedef struct ucc_context_progress_entry {
    ucc_list_link_t            list_elem;
    ucc_context_progress_fn_t  fn;
    void                      *arg;
} ucc_context_progress_entry_t;

ucc_status_t ucc_context_progress_register(ucc_context_t *ctx,
                                           ucc_context_progress_fn_t fn,
                                           void *progress_arg)
{
    ucc_context_progress_entry_t *entry =
        ucc_malloc(sizeof(*entry), "progress_entry");
    if (!entry) {
        ucc_error("failed to allocate %zd bytes for progress entry",
                  sizeof(*entry));
        return UCC_ERR_NO_MEMORY;
    }
    entry->fn  = fn;
    entry->arg = progress_arg;
    ucc_list_add_tail(&ctx->progress_list, &entry->list_elem);
    return UCC_OK;
}

ucc_status_t ucc_context_progress_deregister(ucc_context_t *ctx,
                                             ucc_context_progress_fn_t fn,
                                             void *progress_arg)
{
    ucc_context_progress_entry_t *entry, *tmp;
    ucc_list_for_each_safe(entry, tmp, &ctx->progress_list, list_elem) {
        if (entry->fn == fn && entry->arg == progress_arg) {
            ucc_list_del(&entry->list_elem);
            ucc_free(entry);
            return UCC_OK;
        }
    }
    return UCC_ERR_NOT_FOUND;
}

ucc_status_t ucc_context_progress(ucc_context_h context)
{
    static int                    call_num = 0;
    ucc_status_t                  status;
    ucc_context_progress_entry_t *entry;
    int                           is_empty;

    is_empty = ucc_progress_queue_is_empty(context->pq);
    if (ucc_likely(is_empty)) {
        call_num--;
        if (ucc_likely(call_num >= 0)) {
            return UCC_OK;
        }
        /* progress registered progress fns */
        ucc_list_for_each(entry, &context->progress_list, list_elem) {
            entry->fn(entry->arg);
        }
        call_num = context->throttle_progress;
        return UCC_OK;
    }

    /* the fn below returns int - number of completed tasks.
       TODO : do we need to handle it ? Maybe return to user
       as int as well? */
    status = (ucc_status_t)ucc_progress_queue(context->pq);
    return (status >= 0 ? UCC_OK : status);
}

static ucc_status_t ucc_context_pack_addr(ucc_context_t             *context,
                                          ucc_context_addr_len_t    *addr_len,
                                          int                       *n_packed,
                                          ucc_context_addr_header_t *h)
{
    ucc_context_addr_len_t total_len = 0;
    ptrdiff_t              offset    = 0;
    int                    i, packed, last_packed;
    ucc_base_ctx_attr_t    attr;
    ucc_cl_lib_t          *cl_lib;
    ucc_tl_lib_t          *tl_lib;
    ucc_status_t           status;


    attr.attr.mask = UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN;
    if (h) {
        attr.attr.mask |= UCC_CONTEXT_ATTR_FIELD_CTX_ADDR;
        offset      = (ptrdiff_t)UCC_CONTEXT_ADDR_DATA(h) - (ptrdiff_t)h;
        last_packed = 0;
    }
    packed = 0;

    for (i = 0; i < context->n_cl_ctx; i++) {
        cl_lib = ucc_derived_of(context->cl_ctx[i]->super.lib, ucc_cl_lib_t);
        attr.attr.ctx_addr = PTR_OFFSET(h, offset);
        status =
            cl_lib->iface->context.get_attr(&context->cl_ctx[i]->super, &attr);
        if (UCC_OK != status) {
            ucc_error("failed to query addr len from %s",
                      cl_lib->super.log_component.name);
            return status;
        }
        total_len += attr.attr.ctx_addr_len;
        packed++;
        if (h) {
            h->components[last_packed].id     = cl_lib->iface->super.id;
            h->components[last_packed].offset = offset;
            last_packed++;
            offset += attr.attr.ctx_addr_len;
        }
    }

    for (i = 0; i < context->n_tl_ctx; i++) {
        tl_lib = ucc_derived_of(context->tl_ctx[i]->super.lib, ucc_tl_lib_t);
        attr.attr.ctx_addr = PTR_OFFSET(h, offset);
        status =
            tl_lib->iface->context.get_attr(&context->tl_ctx[i]->super, &attr);
        if (UCC_OK != status) {
            ucc_error("failed to query addr len from %s",
                      tl_lib->super.log_component.name);
            return status;
        }
        total_len += attr.attr.ctx_addr_len;
        packed++;
        if (h) {
            h->components[last_packed].id     = tl_lib->iface->super.id;
            h->components[last_packed].offset = offset;
            last_packed++;
            offset += attr.attr.ctx_addr_len;
        }
    }

    if (addr_len) {
        *addr_len = total_len + UCC_CONTEXT_ADDR_HEADER_SIZE(packed);
    }

    if (n_packed) {
        *n_packed = packed;
    }
    return UCC_OK;
}

ucc_status_t ucc_context_get_attr(ucc_context_t      *context,
                                  ucc_context_attr_t *context_attr)
{
    ucc_status_t               status   = UCC_OK;
    ucc_context_addr_header_t *h;
    if (context_attr->mask & (UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN |
                              UCC_CONTEXT_ATTR_FIELD_CTX_ADDR)) {
        if (!(context->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN)) {
            /* addrlen is not computed yet - do it once and cache on
               context->attr */
            status = ucc_context_pack_addr(context, &context->attr.ctx_addr_len,
                                           &context->n_addr_packed, NULL);
            if (UCC_OK != status) {
                ucc_error("failed to calc ucc context address length");
                return status;
            }
            context->attr.mask |= UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN;
        }
        context_attr->ctx_addr_len = context->attr.ctx_addr_len;
    }

    if (context_attr->mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR) {
        if (!(context->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR)) {
            /* addr_len and n_packed are computed already */
            h = ucc_malloc(context->attr.ctx_addr_len, "ucc_context_address");
            if (!h) {
                ucc_error("failed to allocate %zd bytes for ucc_context_address",
                          context->attr.ctx_addr_len);
                return UCC_ERR_NO_MEMORY;
            }
            h->ctx_id       = context->id;
            h->n_components = context->n_addr_packed;
            status          = ucc_context_pack_addr(context, NULL, NULL, h);
            if (UCC_OK != status) {
                ucc_error("failed to calc ucc context address length");
                return status;
            }
            context->attr.mask    |= UCC_CONTEXT_ATTR_FIELD_CTX_ADDR;
            context->attr.ctx_addr = (ucc_context_addr_h)h;
        }
        context_attr->ctx_addr = context->attr.ctx_addr;
    }

    if (context_attr->mask & UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE) {
        uint64_t            max_buffer_size = 0;
        int                 i;
        ucc_base_ctx_attr_t attr;
        ucc_tl_lib_t *      tl_lib;

        memset(&attr.attr, 0, sizeof(ucc_context_attr_t));
        attr.attr.mask = UCC_CONTEXT_ATTR_FIELD_WORK_BUFFER_SIZE;
        attr.attr.global_work_buffer_size = 0;
        for (i = 0; i < context->n_tl_ctx; i++) {
            tl_lib =
                ucc_derived_of(context->tl_ctx[i]->super.lib, ucc_tl_lib_t);
            status = tl_lib->iface->context.get_attr(&context->tl_ctx[i]->super,
                                                     &attr);
            if (UCC_OK != status) {
                ucc_error("failed to obtain global work buffer size");
                return status;
            }
            if (attr.attr.global_work_buffer_size > max_buffer_size) {
                max_buffer_size = attr.attr.global_work_buffer_size;
            }
        }
        context_attr->global_work_buffer_size = max_buffer_size;
    }

    return status;
}
