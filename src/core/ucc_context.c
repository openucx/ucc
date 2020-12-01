/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_context.h"
#include "cl/ucc_cl.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "cl/ucc_cl_type.h"
ucc_status_t ucc_context_config_read(ucc_lib_info_t *lib, const char *filename,
                                     ucc_context_config_t **config_p)
{
    int                   i;
    ucc_status_t          status;
    ucc_context_config_t *config;

    config = (ucc_context_config_t *)ucc_malloc(sizeof(ucc_context_config_t),
                                                "ctx_config");
    if (config == NULL) {
        ucc_error("failed to allocate %zd bytes for context config",
                  sizeof(ucc_context_config_t));
        status = UCC_ERR_NO_MEMORY;
        goto err_config;
    }

    config->configs = (ucc_cl_context_config_t **)ucc_calloc(
        lib->n_libs_opened, sizeof(ucc_cl_context_config_t *),
        "cl_configs_array");
    if (config->configs == NULL) {
        ucc_error("failed to allocate %zd bytes for cl configs array",
                  sizeof(ucc_cl_context_config_t *));
        status = UCC_ERR_NO_MEMORY;
        goto err_configs;
    }

    config->n_cl_cfg = 0;
    for (i = 0; i < lib->n_libs_opened; i++) {
        ucc_assert(NULL != lib->libs[i]->iface->cl_context_config.table);
        config->configs[i] = (ucc_cl_context_config_t *)ucc_malloc(
            lib->libs[i]->iface->cl_context_config.size, "cl_config");
        if (!config->configs[i]) {
            ucc_error("failed to allocate %zd bytes for cl config",
                      sizeof(lib->libs[i]->iface->cl_context_config.size));
            status = UCC_ERR_NO_MEMORY;
            goto err_config_i;
        }
        status = ucc_config_parser_fill_opts(
            config->configs[config->n_cl_cfg],
            lib->libs[i]->iface->cl_context_config.table, lib->full_prefix,
            lib->libs[i]->iface->cl_context_config.prefix, 0);
        if (UCC_OK != status) {
            ucc_error("failed to read CL \"%s\" context configuration",
                      lib->libs[i]->iface->super.name);
            free(config->configs[i]);
            goto err_config_i;
        }
        config->configs[config->n_cl_cfg]->iface  = lib->libs[i]->iface;
        config->configs[config->n_cl_cfg]->cl_lib = lib->libs[i];
        config->n_cl_cfg++;
    }
    config->lib = lib;
    *config_p   = config;
    return UCC_OK;

err_config_i:
    for (i = i - 1; i >= 0; i--) {
        free(config->configs[i]);
    }
err_configs:
    free(config->configs);

err_config:
    free(config);
    return status;
}

/* Look up the cl_context_config in the array of configs based on the
   cl_type. returns NULL if not found */
static inline ucc_cl_context_config_t *
find_cl_context_config(ucc_context_config_t *cfg, ucc_cl_type_t cl_type)
{
    int i;
    for (i = 0; i < cfg->n_cl_cfg; i++) {
        if (cfg->configs[i] && cl_type == cfg->configs[i]->iface->type) {
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
                cl_cfg, cl_cfg->iface->cl_context_config.table, name, value);
            if (UCC_OK != status) {
                ucc_error("failed to modify CL \"%s\" configuration, name %s, "
                          "value %s",
                          cl_cfg->iface->super.name, name, value);
                return status;
            }
        }
        free(required_cls);
    } else {
        for (i = 0; i < config->n_cl_cfg; i++) {
            if (config->configs[i]) {
                status = ucc_config_parser_set_value(
                    config->configs[i],
                    config->lib->libs[i]->iface->cl_context_config.table, name,
                    value);
                if (UCC_OK != status) {
                    ucc_error("failed to modify CL \"%s\" configuration, name "
                              "%s, value %s",
                              config->lib->libs[i]->iface->super.name, name,
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
        ucc_config_parser_release_opts(
            config->configs[i],
            config->lib->libs[i]->iface->cl_context_config.table);
        free(config->configs[i]);
    }
    free(config->configs);
    free(config);
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
                config->lib->libs[i]->iface->cl_context_config.table,
                config->lib->libs[i]->iface->cl_context_config.prefix,
                config->lib->full_prefix, UCC_CONFIG_PRINT_HEADER);
        }

        ucc_config_parser_print_opts(
            stream, config->lib->libs[i]->iface->cl_context_config.name,
            config->configs[i],
            config->lib->libs[i]->iface->cl_context_config.table,
            config->lib->libs[i]->iface->cl_context_config.prefix,
            config->lib->full_prefix, (ucc_config_print_flags_t)flags);
    }
}
