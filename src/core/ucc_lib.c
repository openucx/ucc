/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"

#include "ucc_global_opts.h"
#include "api/ucc.h"
#include "api/ucc_status.h"
#include "ucc_lib.h"
#include "utils/ucc_log.h"
#include "utils/ucc_malloc.h"

static ucc_config_field_t ucc_lib_config_table[] = {
    {"CLS", "all", "Comma separated list of CL components to be used",
     ucc_offsetof(ucc_lib_config_t, cls), UCC_CONFIG_TYPE_STRING_ARRAY},

    {NULL}
};

ucc_status_t ucc_init_version(unsigned api_major_version,
                              unsigned api_minor_version,
                              const ucc_lib_params_t *params,
                              const ucc_lib_config_h *config,
                              ucc_lib_h *lib_p)
{
    unsigned major_version, minor_version, release_number;
    ucc_status_t status;

    if (UCC_OK != (status = ucc_constructor())) {
        return status;
    }

    ucc_get_version(&major_version, &minor_version, &release_number);

    if ((api_major_version != major_version) ||
        ((api_major_version == major_version) && (api_minor_version > minor_version))) {
        ucc_warn("UCC version is incompatible, required: %d.%d, actual: %d.%d.%d",
                  api_major_version, api_minor_version,
                  major_version, minor_version, release_number);
    }

    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_lib_config_read(const char *env_prefix, const char *filename,
                                 ucc_lib_config_t **config_p)
{
    ucc_lib_config_t *config;
    ucc_status_t      status;
    size_t            full_prefix_len;
    const char       *base_prefix = "UCC_";

    config = ucc_malloc(sizeof(*config), "lib_config");
    if (config == NULL) {
        ucc_error("failed to allocate %zd bytes for lib_config",
                  sizeof(*config));
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }
    /* full_prefix for a UCC lib config is either just
          (i)  "UCC_" - if no "env_prefix" is provided, or
          (ii) "AAA_UCC" - where AAA is the value of "env_prefix".
       Allocate space to build prefix str and two characters ("_" and “\0”) */
    full_prefix_len =
        strlen(base_prefix) + (env_prefix ? strlen(env_prefix) : 0) + 2;
    config->full_prefix = ucc_malloc(full_prefix_len, "full_prefix");
    if (!config->full_prefix) {
        ucc_error("failed to allocate %zd bytes for full_prefix",
                  full_prefix_len);
        status = UCC_ERR_NO_MEMORY;
        goto err_free_config;
    }
    if (env_prefix) {
        ucc_snprintf_safe(config->full_prefix, full_prefix_len, "%s_%s",
                          env_prefix, base_prefix);
    } else {
        ucc_strncpy_safe(config->full_prefix, base_prefix,
                         strlen(base_prefix) + 1);
    }

    status = ucc_config_parser_fill_opts(config, ucc_lib_config_table,
                                         config->full_prefix, NULL, 0);
    if (status != UCC_OK) {
        ucc_error("failed to read UCC lib config");
        goto err_free_prefix;
    }
    *config_p = config;
    return UCC_OK;

err_free_prefix:
    free(config->full_prefix);
err_free_config:
    free(config);
err:
    return status;
}

void ucc_lib_config_release(ucc_lib_config_t *config)
{
    ucc_config_parser_release_opts(config, ucc_lib_config_table);
    free(config->full_prefix);
    free(config);
}

void ucc_lib_config_print(const ucc_lib_config_h config, FILE *stream,
                          const char *title, ucc_config_print_flags_t print_flags)
{
    ucc_config_parser_print_opts(stream, title, config, ucc_lib_config_table,
                                 NULL, config->full_prefix, print_flags);
}

ucc_status_t ucc_lib_config_modify(ucc_lib_config_h config, const char *name,
                                   const char *value)
{
    return ucc_config_parser_set_value(config, ucc_lib_config_table, name,
                                       value);
}
