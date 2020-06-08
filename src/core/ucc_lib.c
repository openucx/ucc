#include <ucc_lib.h>
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include "utils/ucc_log.h"
#include <team_lib/ucc_tl.h>

static ucs_config_field_t ucc_lib_config_table[] = {
    {"TLS", "all",
     "Comma separated list of TeamLibrarieS to be used",
     ucs_offsetof(ucc_lib_config_t, tls),
     UCS_CONFIG_TYPE_STRING_ARRAY},

    {NULL}
};

UCS_CONFIG_REGISTER_TABLE(ucc_lib_config_table, "UCC", NULL, ucc_lib_config_t)

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{                       \
        if ((params->field_mask & UCC_LIB_PARAM_FIELD_ ## _CAP_FIELD) && \
            !(params-> _cap & tl_iface->params. _cap)) {                 \
            ucc_info("Disqualifying team %s due to %s cap",              \
                     tl_iface->name, UCS_PP_QUOTE(_CAP_FIELD));          \
            continue;                                                    \
        }                                                                \
    } while(0)

static inline ucc_status_t ucc_tl_is_loaded(char *tl_name)
{
    int i;
    for (i=0; i<ucc_lib_data.n_tls_loaded; i++) {
        if (0 == strncmp(tl_name, ucc_lib_data.tl_ifaces[i]->name,
                         strlen(tl_name))) {
            return UCC_OK;
        }
    }
    return UCC_ERR_NO_MESSAGE;
}

static inline ucc_status_t ucc_lib_config_tls_check(ucc_lib_t *lib,
                                                    ucs_config_names_array_t *tls)
{
    int i;
    if (tls->count == 0) {
        return UCC_ERR_NO_MESSAGE;
    }
    if (tls->count == 1 && 0 == strcmp(tls->names[0], "all")) {
        return UCC_OK;
    }
    for (i=0; i<tls->count; i++) {
        if (UCS_OK != ucc_tl_is_loaded(tls->names[i])) {
            ucc_error("Required TL: \"%s\" (ucc_team_lib_%s.so) is not available\n",
                      tls->names[i], tls->names[i]);
            return UCC_ERR_NO_MESSAGE;
        }
    }
    return UCC_OK;
}


static ucc_status_t ucc_lib_init_filtered(const ucc_lib_params_t *params,
                                          const ucc_lib_config_t *config,
                                          ucc_lib_t *lib)
{
    int n_tls = ucc_lib_data.n_tls_loaded;
    ucc_tl_iface_t *tl_iface;
    ucc_team_lib_t *tl_lib;
    ucc_tl_lib_config_t *tl_config;
    ucc_status_t status;
    int i, need_tl_name_check;

    lib->libs = (ucc_team_lib_t**)malloc(sizeof(ucc_team_lib_t*)*n_tls);
    if (!lib->libs) {
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    assert(config->tls.count >= 1);
    need_tl_name_check = (0 != strcmp(config->tls.names[0], "all"));
    lib->n_libs_opened = 0;
    for (i=0; i<n_tls; i++) {
        tl_iface = ucc_lib_data.tl_ifaces[i];
        if (need_tl_name_check &&
            -1 == ucs_config_names_search(config->tls, tl_iface->name)) {
            continue;
        }
        CHECK_LIB_CONFIG_CAP(reproducible, REPRODUCIBLE);
        CHECK_LIB_CONFIG_CAP(thread_mode,  THREAD_MODE);
        CHECK_LIB_CONFIG_CAP(coll_types,   COLL_TYPES);
        tl_config  = malloc(tl_iface->tl_lib_config.size);
        ucs_config_parser_fill_opts(tl_config, tl_iface->tl_lib_config.table,
                                    config->full_prefix, tl_iface->tl_lib_config.prefix, 0);
        status = tl_iface->init(params, config, tl_config, &tl_lib);

        if (UCS_OK != status) {
            ucs_config_parser_release_opts(tl_config, tl_iface->tl_lib_config.table);
            ucc_error("lib_init failed for TL: %s\n", tl_iface->name);
            goto error;
        }
        tl_lib->log_component = tl_config->log_component;
        tl_lib->priority = (-1 == tl_config->priority) ?
            tl_iface->priority : tl_config->priority;
        ucs_config_parser_release_opts(tl_config, tl_iface->tl_lib_config.table);
        lib->libs[lib->n_libs_opened++] = tl_lib;
        ucc_info("lib_prefix \"%s\": initialized tl \"%s\" priority %d\n",
                 config->full_prefix, tl_iface->name, tl_lib->priority);
    }
    return UCS_OK;

error:
    if (lib->libs) free(lib->libs);
    return status;
}

ucc_status_t ucc_lib_init(const ucc_lib_params_t *params,
                          const ucc_lib_config_h config,
                          ucc_lib_h *ucc_lib)
{
    ucs_status_t status;
    ucc_lib_t *lib;

    if (ucc_lib_data.n_tls_loaded == 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    lib = malloc(sizeof(*lib));
    if (!lib) {
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }

    status = ucc_lib_config_tls_check(lib, &config->tls);
    if (UCS_OK != status) {
        ucc_error("Unsupported \"UCC_TLS\" value\n");
        goto error;
    }
    status = ucc_lib_init_filtered(params, (const ucc_lib_config_t*)config, lib);
    if (UCS_OK != status) {
        goto error;
    }

    if (lib->n_libs_opened == 0) {
        ucc_error("UCC lib init: no plugins left after filtering by params\n");
        status = UCC_ERR_NO_MESSAGE;
        goto error;
    }

    *ucc_lib = lib;
    //TODO: move to appropriate place
    //ucs_config_parser_warn_unused_env_vars_once("UCC_");
    return UCC_OK;

error:
    *ucc_lib = NULL;
    if (lib) free(lib);
    return status;
}

ucc_status_t ucc_lib_config_read(const char *env_prefix,
                                 const char *filename,
                                 ucc_lib_config_t **config_p){
    ucc_lib_config_t *config;
    ucc_status_t status;
    int full_prefix_len;
    const char* base_prefix = "UCC_";

    config = malloc(sizeof(*config));
    if (config == NULL) {
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }
    full_prefix_len = strlen(base_prefix)  +
        (env_prefix ? strlen(env_prefix) : 0) + 2;
    config->full_prefix = malloc(full_prefix_len);
    if (env_prefix) {
        snprintf(config->full_prefix, full_prefix_len,
                 "%s_%s", env_prefix, base_prefix);
    } else {
        strcpy(config->full_prefix, base_prefix);
    }

    status = ucs_config_parser_fill_opts(config, ucc_lib_config_table,
                                         config->full_prefix, NULL, 0);
    if (status != UCS_OK) {
        goto err_free;
    }
    *config_p = config;
    return UCC_OK;

err_free:
    free(config);
err:
    return status;
}

void ucc_lib_config_release(ucc_lib_config_t *config)
{
    free(config->full_prefix);
    free(config);
}

void ucc_lib_config_print(const ucc_lib_config_t *config, FILE *stream,
                          const char *title, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, ucc_lib_config_table,
                                 NULL, "UCC_", print_flags);
}

void ucc_lib_cleanup(ucc_lib_t *lib)
{
    int i;
    assert(lib->n_libs_opened > 0);
    assert(lib->libs);
    for (i=0; i<lib->n_libs_opened; i++) {
        lib->libs[i]->iface->cleanup(lib->libs[i]);
    }
    free(lib->libs);
    free(lib);
}
