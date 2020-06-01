#include <ucc_lib.h>
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include "utils/ucc_log.h"
#include <team_lib/ucc_tl.h>

extern ucc_lib_t ucc_static_lib;
static ucs_config_field_t ucc_lib_config_table[] = {
    {NULL}
};

UCS_CONFIG_REGISTER_TABLE(ucc_lib_config_table, "UCC", NULL, ucc_lib_config_t)

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{                       \
        if ((params->field_mask & UCC_LIB_PARAM_FIELD_ ## _CAP_FIELD) && \
            !(params-> _cap & tl->params. _cap)) {                       \
            ucc_info("Disqualifying team %s due to %s cap",              \
                     tl->name, UCS_PP_QUOTE(_CAP_FIELD));                \
            continue;                                                    \
        }                                                                \
    } while(0)


static void ucc_lib_filter(const ucc_lib_params_t *params, ucc_lib_t *lib)
{
    int i;
    int n_libs = ucc_static_lib.n_libs_opened;
    lib->libs = (ucc_team_lib_t**)malloc(sizeof(ucc_team_lib_t*)*n_libs);
    lib->n_libs_opened = 0;
    for (i=0; i<n_libs; i++) {
        ucc_team_lib_t *tl = ucc_static_lib.libs[i];
        CHECK_LIB_CONFIG_CAP(reproducible, REPRODUCIBLE);
        CHECK_LIB_CONFIG_CAP(thread_mode,  THREAD_MODE);
        CHECK_LIB_CONFIG_CAP(coll_types,   COLL_TYPES);
        lib->libs[lib->n_libs_opened++] = tl;
    }
}

ucc_status_t ucc_lib_init(const ucc_lib_params_t *params,
                          const ucc_lib_config_t *config,
                          ucc_lib_h *ucc_lib)
{
    ucc_lib_t *lib;

    if (ucc_static_lib.n_libs_opened == 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    lib = malloc(sizeof(*lib));
    if (lib == NULL) {
        return UCC_ERR_NO_MEMORY;
    }

    ucc_lib_filter(params, lib);
    if (lib->n_libs_opened == 0) {
        ucc_error("UCC lib init: no plugins left after filtering by params\n");
        return UCC_ERR_NO_MESSAGE;
    }

    *ucc_lib = lib;
    //TODO: move to appropriate place
    //ucs_config_parser_warn_unused_env_vars_once("UCC_");
    return UCC_OK;
}

ucc_status_t ucc_lib_config_read(const char *env_prefix,
                                 const char *filename,
                                 ucc_lib_config_t **config_p){
    ucc_lib_config_t *config;
    ucc_status_t status;
    char full_prefix[128] = "UCC_";

    config = malloc(sizeof(*config));
    if (config == NULL) {
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }

    if ((env_prefix != NULL) && (strlen(env_prefix) > 0)) {
        snprintf(full_prefix, sizeof(full_prefix), "%s_%s", env_prefix, "UCC_");
    }

    status = ucs_config_parser_fill_opts(config, ucc_lib_config_table, full_prefix,
                                         NULL, 0);
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
    free(config);
}

void ucc_lib_config_print(const ucc_lib_config_t *config, FILE *stream,
                          const char *title, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, ucc_lib_config_table,
                                 NULL, "UCC_", print_flags);
}

void ucc_lib_cleanup(ucc_lib_h lib_p)
{
    if (lib_p->libs) {
        free(lib_p->libs);
    }
    free(lib_p);
}
