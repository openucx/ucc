#include <ucc_lib.h>
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include "utils/ucc_log.h"
#include "ccm/ucc_ccm.h"

static ucs_config_field_t ucc_lib_config_table[] = {
    {"COLL_COMPONENTS", "all",
     "Comma separated list of coll components to be used",
     ucs_offsetof(ucc_lib_config_t, ccms),
     UCS_CONFIG_TYPE_STRING_ARRAY},

    {NULL}
};

UCS_CONFIG_REGISTER_TABLE(ucc_lib_config_table, "UCC", NULL, ucc_lib_config_t)

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{                       \
        if ((params->mask & UCC_LIB_PARAM_FIELD_ ## _CAP_FIELD) &&       \
            !(params-> _cap & ccm_iface->params. _cap)) {                \
            ucc_info("Disqualifying component %s due to %s cap",         \
                     ccm_iface->name, UCS_PP_QUOTE(_CAP_FIELD));         \
            continue;                                                    \
        }                                                                \
    } while(0)

static inline ucc_status_t ucc_component_is_loaded(char *ccm_name)
{
    int i;
    for (i=0; i<ucc_lib_data.n_ccms_loaded; i++) {
        if (0 == strncmp(ccm_name, ucc_lib_data.ccm_ifaces[i]->name,
                         strlen(ccm_name))) {
            return UCC_OK;
        }
    }
    return UCC_ERR_NO_MESSAGE;
}

static inline ucc_status_t ucc_lib_config_components_check(ucc_lib_info_t *lib,
                                                           ucs_config_names_array_t *ccms)
{
    int i;
    if (ccms->count == 0) {
        return UCC_ERR_NO_MESSAGE;
    }
    if (ccms->count == 1 && 0 == strcmp(ccms->names[0], "all")) {
        return UCC_OK;
    }
    for (i=0; i<ccms->count; i++) {
        if (UCS_OK != ucc_component_is_loaded(ccms->names[i])) {
            ucc_error("Required TL: \"%s\" (ucc_ccm_%s.so) is not available\n",
                      ccms->names[i], ccms->names[i]);
            return UCC_ERR_NO_MESSAGE;
        }
    }
    return UCC_OK;
}


static ucc_status_t ucc_lib_init_filtered(const ucc_lib_params_t *params,
                                          const ucc_lib_config_t *config,
                                          ucc_lib_info_t *lib)
{
    int n_ccms = ucc_lib_data.n_ccms_loaded;
    ucc_ccm_iface_t *ccm_iface;
    ucc_ccm_lib_t   *ccm_lib;
    ucc_ccm_lib_config_t *ccm_config;
    ucc_status_t status;
    int i, need_ccm_name_check;

    lib->libs = (ucc_ccm_lib_t**)malloc(sizeof(ucc_ccm_lib_t*)*n_ccms);
    if (!lib->libs) {
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    assert(config->ccms.count >= 1);
    need_ccm_name_check = (0 != strcmp(config->ccms.names[0], "all"));
    lib->n_libs_opened = 0;
    for (i=0; i<n_ccms; i++) {
        ccm_iface = ucc_lib_data.ccm_ifaces[i];
        if (need_ccm_name_check &&
            -1 == ucs_config_names_search(config->ccms, ccm_iface->name)) {
            continue;
        }
        CHECK_LIB_CONFIG_CAP(thread_mode,  THREAD_MODE);
        CHECK_LIB_CONFIG_CAP(coll_types,   COLL_TYPES);
        ccm_config  = malloc(ccm_iface->ccm_lib_config.size);
        ucs_config_parser_fill_opts(ccm_config, ccm_iface->ccm_lib_config.table,
                                    config->full_prefix, ccm_iface->ccm_lib_config.prefix, 0);
        status = ccm_iface->init(params, config, ccm_config, &ccm_lib);

        if (UCS_OK != status) {
            ucs_config_parser_release_opts(ccm_config, ccm_iface->ccm_lib_config.table);
            ucc_error("lib_init failed for component: %s\n", ccm_iface->name);
            goto error;
        }
        ccm_lib->log_component = ccm_config->log_component;
        snprintf(ccm_lib->log_component.name, strlen(ccm_iface->ccm_lib_config.prefix),
                 "%s", ccm_iface->ccm_lib_config.prefix);
        ccm_lib->priority = (-1 == ccm_config->priority) ?
            ccm_iface->priority : ccm_config->priority;
        ucs_config_parser_release_opts(ccm_config, ccm_iface->ccm_lib_config.table);
        lib->libs[lib->n_libs_opened++] = ccm_lib;
        ucc_info("lib_prefix \"%s\": initialized component \"%s\" priority %d\n",
                 config->full_prefix, ccm_iface->name, ccm_lib->priority);
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
    ucc_lib_info_t *lib;

    if (ucc_lib_data.n_ccms_loaded == 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    lib = malloc(sizeof(*lib));
    if (!lib) {
        status = UCC_ERR_NO_MEMORY;
        goto error;
    }
    lib->full_prefix= strdup(config->full_prefix);

    status = ucc_lib_config_components_check(lib, &config->ccms);
    if (UCS_OK != status) {
        ucc_error("Unsupported \"UCC_COLL_COMPONENTS\" value\n");
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

void ucc_lib_config_print(const ucc_lib_config_h config, FILE *stream,
                          const char *title, ucc_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, ucc_lib_config_table,
                                 NULL, "UCC_", print_flags);
}

void ucc_lib_cleanup(ucc_lib_info_t *lib)
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
