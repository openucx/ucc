/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include "ucc_context.h"
#include "team_lib/ucc_tl.h"
#include <ucs/sys/math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


ucc_status_t ucc_context_create(ucc_lib_h lib,
                                const ucc_context_params_t *params,
                                const ucc_context_config_h config,
                                ucc_context_h *context)
{
    ucc_context_t    *ctx;
    uint64_t         i;
    ucc_team_lib_t   *tl_lib;
    ucc_tl_context_t *tl_ctx;
    ucc_status_t     status;
    int              num_tls;

    num_tls = config->n_tl_cfg;
    ctx = malloc(sizeof(ucc_context_t));
    if (!ctx) {
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    ctx->lib = lib;
    memcpy(&ctx->params, params, sizeof(ucc_context_params_t));

    ctx->tl_ctx = (ucc_tl_context_t**)malloc(sizeof(ucc_tl_context_t*)*num_tls);
    if (!ctx->tl_ctx) {
        status = UCC_ERR_NO_MEMORY;
        goto error_ctx;
    }
    ctx->n_tl_ctx = num_tls;

    for (i=0; i<num_tls; i++) {
        tl_lib = config->configs[i]->tl_lib;
        status = config->configs[i]->iface->context_create(tl_lib, params,
                                                           config->configs[i], &tl_ctx);
        tl_ctx->tl_lib = tl_lib;
        ctx->tl_ctx[i] = tl_ctx;
    }

    *context = ctx;
    return UCC_OK;

error_ctx:
    free(ctx);
error:
    return status;
}

ucc_status_t ucc_context_progress(ucc_context_h context)
{
    //TODO
    return UCC_OK;
}

void ucc_context_destroy(ucc_context_h context)
{
    ucc_tl_context_t *tl_ctx;
    int              i;

    for (i = 0; i < context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        tl_ctx->tl_lib->iface->context_destroy(tl_ctx);
    }

    free(context->tl_ctx);
    free(context);
}

ucc_status_t ucc_context_config_read(ucc_lib_t *lib,
                                     ucc_context_config_t **config_p)
{
    int               i;
    ucs_status_t status;
    ucc_context_config_t *config;

    config = (ucc_context_config_t*)malloc(sizeof(ucc_context_config_t));
    if (config == NULL) {
        status = UCC_ERR_NO_MEMORY;
        goto err_config;
    }

    config->configs = (ucc_tl_context_config_t**)calloc(lib->n_libs_opened,
                                                        sizeof(ucc_tl_context_config_t*));
    if (config->configs == NULL) {
        status = UCC_ERR_NO_MEMORY;
        goto err_configs;
    }

    config->n_tl_cfg = 0;
    //TODO parse UCC_TLS set in config
    for(i = 0; i < lib->n_libs_opened; i++) {
        assert(NULL != lib->libs[i]->iface->tl_context_config.table);
        config->configs[i] = (ucc_tl_context_config_t*)
            malloc(lib->libs[i]->iface->tl_context_config.size);
        if (!config->configs[i]) {
            status = UCC_ERR_NO_MEMORY;
            goto err_config_i;
        }

        status = ucs_config_parser_fill_opts(config->configs[config->n_tl_cfg],
                                             lib->libs[i]->iface->tl_context_config.table,
                                             lib->full_prefix,
                                             lib->libs[i]->iface->tl_context_config.prefix,
                                             0);
        config->configs[config->n_tl_cfg]->iface  = lib->libs[i]->iface;
        config->configs[config->n_tl_cfg]->tl_lib = lib->libs[i];
        config->n_tl_cfg++;
        //TODO check status
    }
    config->lib      = lib;
    *config_p = config;
    return UCC_OK;

err_config_i:
    for(i = i - 1;i >= 0; i--) {
        free(config->configs[i]);
    }
err_configs:
    free(config->configs);

err_config:
    free(config);
    return status;
}

ucc_status_t ucc_context_config_modify(ucc_context_config_t *config,
                                       const char *name, const char *value)
{
    int i;
    for(i = 0; i < config->n_tl_cfg; i++) {
        if (config->configs[i]) {
            ucs_config_parser_set_value(config->configs[i],
                                        config->lib->libs[i]->iface->tl_context_config.table,
                                        name, value);
        }
    }
    return UCC_OK;
}

void ucc_context_config_release(ucc_context_config_t *config)
{
    int i;
    for(i = 0; i < config->n_tl_cfg; i++) {
        if (!config->configs[i]) {
            continue;
        }
        ucs_config_parser_release_opts(config->configs[i],
                                       config->lib->libs[i]->iface->tl_context_config.table);
        free(config->configs[i]);
    }
    free(config->configs);
    free(config);
}
