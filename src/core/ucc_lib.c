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
#include "utils/ucc_math.h"
#include "cl/ucc_cl.h"
#include "ucp_ctx/ucc_ucp_ctx.h"

UCS_CONFIG_DEFINE_ARRAY(cl_types, sizeof(ucc_cl_type_t),
                        UCS_CONFIG_TYPE_ENUM(ucc_cl_names));

static ucc_config_field_t ucc_lib_config_table[] = {
    {"CLS", "all", "Comma separated list of CL components to be used",
     ucc_offsetof(ucc_lib_config_t, cls), UCC_CONFIG_TYPE_ARRAY(cl_types)},

    {NULL}
};

UCC_CONFIG_REGISTER_TABLE(ucc_lib_config_table, "UCC", NULL, ucc_lib_config_t,
                          &ucc_config_global_list)

static inline ucc_status_t ucc_cl_component_is_loaded(ucc_cl_type_t cl_type)
{
    const char *cl_name = ucc_cl_names[cl_type];
    if (NULL == ucc_get_component(&ucc_global_config.cl_framework, cl_name)) {
        return UCC_ERR_NOT_FOUND;
    } else {
        return UCC_OK;
    }
}

static inline int ucc_cl_requested(const ucc_lib_config_t *cfg, int cl_type)
{
    int i;
    for (i = 0; i < cfg->cls.count; i++) {
        if (cfg->cls.types[i] == cl_type) {
            return 1;
        }
    }
    return 0;
}

/* Core logic for the selection of CL components:
   1. If user does not provide a set of required CLs then we try to
   use only those components that support the required input params.
   This means, if some component does not pass params check it is skipped.

   2. In contrast, if user explicitly requests a list of CLs to use,
   then we try to load ALL of them and report the supported attributes
   based on that selection. */
static ucc_status_t ucc_lib_init_filtered(const ucc_lib_params_t *user_params,
                                          const ucc_lib_config_t *config,
                                          ucc_lib_info_t *lib)
{
    int                  n_cls = ucc_global_config.cl_framework.n_components;
    uint64_t             supported_coll_types = 0;
    ucc_thread_mode_t    supported_tm         = UCC_THREAD_MULTIPLE;
    ucc_lib_params_t     params               = *user_params;
    ucc_cl_iface_t      *cl_iface;
    ucc_cl_lib_t        *cl_lib;
    ucc_cl_lib_config_t *cl_config;
    ucc_status_t         status;
    int                  i, specific_cls_requested;

    lib->libs =
        (ucc_cl_lib_t **)ucc_malloc(sizeof(ucc_cl_lib_t *) * n_cls, "cl_libs");
    if (!lib->libs) {
        ucc_error("failed to allocate %zd bytes for cl_libs",
                  sizeof(ucc_cl_lib_t *) * n_cls);
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    if (!(params.mask & UCC_LIB_PARAM_FIELD_THREAD_MODE)) {
        params.mask |= UCC_LIB_PARAM_FIELD_THREAD_MODE;
        params.thread_mode = UCC_THREAD_SINGLE;
    }
    ucc_assert(config->cls.count >= 1);
    specific_cls_requested = (0 == ucc_cl_requested(config, UCC_CL_ALL));
    lib->n_libs_opened     = 0;
    for (i = 0; i < n_cls; i++) {
        cl_iface = ucc_derived_of(ucc_global_config.cl_framework.components[i],
                                  ucc_cl_iface_t);
        /* User requested specific list of CLs and current cl_iface is not part
           of the list: skip it. */
        if (specific_cls_requested &&
            0 == ucc_cl_requested(config, cl_iface->type)) {
            continue;
        }
        if (params.thread_mode > cl_iface->attr.thread_mode) {
            /* Requested THREAD_MODE is not supported by the CL:
               1. If cls == "all" - just skip this CL
               2. If specific CLs are requested: continue and user will 
                  have to query result attributes and check thread mode*/
            if (!specific_cls_requested) {
                ucc_info("requested thread_mode is not supported by the CL: %s",
                         cl_iface->super.name);
                continue;
            }
        }
        cl_config = ucc_malloc(cl_iface->cl_lib_config.size, "cl_lib_config");
        if (!cl_config) {
            status = UCC_ERR_NO_MEMORY;
            goto error_cl_cleanup;
        }
        status = ucc_config_parser_fill_opts(
            cl_config, cl_iface->cl_lib_config.table, config->full_prefix,
            cl_iface->cl_lib_config.prefix, 0);
        if (UCC_OK != status) {
            ucc_error("failed to parse CL %s lib configuration",
                      cl_iface->super.name);
            goto error_cl_config_parse;
        }
        status = cl_iface->init(&params, config, cl_config, &cl_lib);
        if (UCC_OK != status) {
            if (specific_cls_requested) {
                ucc_error("lib_init failed for component: %s",
                          cl_iface->super.name);
                goto error_cl_init;
            } else {
                ucc_info("lib_init failed for component: %s, skipping",
                         cl_iface->super.name);
                ucc_config_parser_release_opts(cl_config,
                                               cl_iface->cl_lib_config.table);
                free(cl_config);
                continue;
            }
        }
        ucc_config_parser_release_opts(cl_config,
                                       cl_iface->cl_lib_config.table);
        free(cl_config);
        lib->libs[lib->n_libs_opened++] = cl_lib;
        supported_coll_types |= cl_iface->attr.coll_types;
        if (cl_iface->attr.thread_mode < supported_tm) {
            supported_tm = cl_iface->attr.thread_mode;
        }
        ucc_info("lib_prefix \"%s\": initialized component \"%s\" priority %d",
                 config->full_prefix, cl_iface->super.name, cl_lib->priority);
    }

    if (lib->n_libs_opened == 0) {
        ucc_error("lib_init failed: no CLs left after filtering");
        status = UCC_ERR_NO_MESSAGE;
        goto error;
    }

    /* Check if the combination of the selected CLs provides all the
       requested coll_types: not an error, just print a message if not
       all the colls are supproted */
    if (params.mask & UCC_LIB_PARAM_FIELD_COLL_TYPES &&
        ((params.coll_types & supported_coll_types) != params.coll_types)) {
        ucc_debug("selected set of CLs does not provide all the requested "
                  "coll_types");
    }
    if (params.thread_mode > supported_tm) {
        ucc_debug("selected set of CLs does not provide the requested "
                  "thread_mode");
    }
    lib->attr.coll_types  = supported_coll_types;
    lib->attr.thread_mode = ucc_min(supported_tm, params.thread_mode);
    return UCS_OK;

error_cl_init:
    ucc_config_parser_release_opts(cl_config, cl_iface->cl_lib_config.table);
error_cl_config_parse:
    free(cl_config);
error_cl_cleanup:
    for (i = 0; i < lib->n_libs_opened; i++) {
        lib->libs[i]->iface->finalize(lib->libs[i]);
    }
error:
    if (lib->libs)
        free(lib->libs);
    return status;
}

ucc_status_t ucc_init_version(unsigned api_major_version,
                              unsigned api_minor_version,
                              const ucc_lib_params_t *params,
                              const ucc_lib_config_h config, ucc_lib_h *lib_p)
{
    unsigned        major_version, minor_version, release_number;
    ucc_status_t    status;
    ucc_lib_info_t *lib;

    *lib_p = NULL;

    if (UCC_OK != (status = ucc_constructor())) {
        return status;
    }

    ucc_get_version(&major_version, &minor_version, &release_number);

    if ((api_major_version != major_version) ||
        ((api_major_version == major_version) &&
         (api_minor_version > minor_version))) {
        ucc_warn(
            "UCC version is incompatible, required: %d.%d, actual: %d.%d.%d",
            api_major_version, api_minor_version, major_version, minor_version,
            release_number);
    }

    lib = ucc_malloc(sizeof(ucc_lib_info_t), "lib_info");
    if (!lib) {
        ucc_error("failed to allocate %zd bytes for lib_info",
                  sizeof(ucc_lib_info_t));
        return UCC_ERR_NO_MEMORY;
    }
    lib->full_prefix = strdup(config->full_prefix);
    if (!lib->full_prefix) {
        ucc_error("failed strdup for full_prefix");
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    /* Initialize ucc lib handle using requirements from the user
       provided via params/config and available CLs in the
       CL component framework.

       The lib_p object will contain the array of ucc_cl_lib_t objects
       that are allocated using CL init/finalize interface. */
    status = ucc_lib_init_filtered(params, config, lib);
    if (UCC_OK != status) {
        goto error;
    }

    *lib_p = lib;
    return UCC_OK;
error:
    free(lib);
    return status;
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
                          const char *title,
                          ucc_config_print_flags_t print_flags)
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

ucc_status_t ucc_finalize(ucc_lib_info_t *lib)
{
    int i;
    ucc_status_t status, return_status = UCC_OK;
    ucc_assert(lib->n_libs_opened > 0);
    ucc_assert(lib->libs);
    /* If some CL components fails in finalize we will return
       its failure status to the user, however we will still
       try to continue and finalize other CLs */
    for (i = 0; i < lib->n_libs_opened; i++) {
        status = lib->libs[i]->iface->finalize(lib->libs[i]);
        if (UCC_OK == return_status && UCC_OK != status) {
            return_status = status;
        }
    }
    free(lib->libs);
    free(lib);
    return return_status;
}
