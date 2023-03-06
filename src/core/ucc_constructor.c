/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_global_opts.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_component.h"
#include "utils/ucc_log.h"
#include "utils/ucc_sys.h"
#include "utils/ucc_string.h"
#include "utils/ucc_proc_info.h"
#include "utils/profile/ucc_profile.h"
#include "ucc/api/ucc_version.h"
#include <dlfcn.h>

static ucc_status_t ucc_check_config_file(void)
{
    ucc_global_config_t *cfg                = &ucc_global_config;
    ucc_status_t         status             = UCC_OK;
    const char *         default_share_name = "/share/ucc.conf";
    const char *         default_home_name  = "/ucc.conf";
    const char *         home;
    char *               filename;

    /* First check the UCC_CONFIG_FILE - most precedence */
    if (strlen(cfg->cfg_filename) == 0) {
        /* file configuration disabled */
        return UCC_OK;
    }
    if (0 != strcasecmp(cfg->cfg_filename, "auto")) {
        /* Actual file path is provided */
        status = ucc_parse_file_config(cfg->cfg_filename,
                                       &ucc_global_config.file_cfg);
        if (UCC_ERR_NOT_FOUND == status) {
            /* File ENV value provided but not available */
            ucc_warn("failed to open config file: %s", cfg->cfg_filename);
        }
        return status;
    }

    /* CONFIG_FILE is set to AUTO. Search HOME and install/share */
    if (NULL != (home = getenv("HOME"))) {
        if (UCC_OK != (status = ucc_str_concat(home ,default_home_name,
                                               &filename))) {
            return status;
        }
        status = ucc_parse_file_config(filename, &ucc_global_config.file_cfg);
        ucc_free(filename);
        if (UCC_ERR_NOT_FOUND != status) {
            /* either OK or fatal error, NOT_FOUND means
               no file in HOME - just continue */
            return status;
        }
    }
    /* Finally, try to find config file in the library install/share */
    if (UCC_OK != (status = ucc_str_concat(cfg->install_path,
                                           default_share_name, &filename))) {
        return status;
    }
    status = ucc_parse_file_config(filename, &ucc_global_config.file_cfg);
    ucc_free(filename);
    return status;
}

static ucc_status_t init_lib_paths(void)
{
    const char  *so_path = ucc_sys_get_lib_path();
    ucc_status_t status;
    char        *lib_path;

    if (!so_path) {
        return UCC_ERR_NOT_FOUND;
    }
    status = ucc_sys_dirname(so_path, &lib_path);
    if (UCC_OK != status) {
        return status;
    }

    status = ucc_sys_dirname(lib_path, &ucc_global_config.install_path);
    if (UCC_OK != status) {
        goto out;
    }
    status = ucc_sys_path_join(lib_path, UCC_MODULE_SUBDIR,
                               &ucc_global_config.component_path);
out:
    free(lib_path);
    return status;
}

UCC_CONFIG_REGISTER_TABLE(ucc_global_config_table, "UCC global", NULL,
                          ucc_global_config, &ucc_config_global_list)

ucc_status_t ucc_constructor(void)
{
    ucc_global_config_t *cfg = &ucc_global_config;
    ucc_status_t         status;
    Dl_info              dl_info;
    int                  ret;

    if (!cfg->initialized) {
        cfg->initialized = 1;
        status = ucc_config_parser_fill_opts(
            &ucc_global_config, UCC_CONFIG_GET_TABLE(ucc_global_config_table),
            "UCC_", 1);
        if (UCC_OK != status) {
            ucc_error("failed to parse global options");
            return status;
        }

        if (UCC_OK != (status = init_lib_paths())) {
            ucc_error("failed to init ucc components path");
            return status;
        }

        status = ucc_check_config_file();
        if (UCC_OK != status && UCC_ERR_NOT_FOUND != status) {
            /* bail only in case of real error */
            return status;
        }

        status = ucc_components_load("cl", &cfg->cl_framework);
        if (UCC_OK != status) {
            ucc_error("no CL components were found in the "
                      "ucc modules dir: %s",
                      cfg->component_path);
            return status;
        }
        status = ucc_component_check_scores_uniq(&cfg->cl_framework);
        if (UCC_OK != status) {
            ucc_error("CLs must have distinct uniq default scores");
            return status;
        }
        status = ucc_components_load("tl", &cfg->tl_framework);
        if (UCC_OK != status) {
            /* not critical - some CLs may operate w/o use of TL */
            ucc_debug("no TL components were found in the "
                      "ucc modules dir: %s",
                      cfg->component_path);
        }
        status = ucc_component_check_scores_uniq(&cfg->tl_framework);
        if (UCC_OK != status) {
            ucc_error("TLs must have distinct uniq default scores");
            return status;
        }
        status = ucc_components_load("mc", &cfg->mc_framework);
        if (UCC_OK != status) {
            ucc_error("no memory components were found in the "
                      "ucc modules dir: %s",
                      cfg->component_path);
            return status;
        }
        status = ucc_components_load("ec", &cfg->ec_framework);
        if (status != UCC_OK) {
            if (status == UCC_ERR_NOT_FOUND) {
                ucc_info("no execution components were found in the "
                         "ucc modules dir: %s. "
                         "Triggered operations might not work",
                         cfg->component_path);
            } else {
                ucc_error("failed to load execution components %d (%s)",
                           status, ucc_status_string(status));
                return status;
            }
        }

        if (UCC_OK != ucc_local_proc_info_init()) {
            ucc_error("failed to initialize local proc info");
            return status;
        }
#ifdef HAVE_PROFILING
        ucc_profile_init(cfg->profile_mode, cfg->profile_file,
                         cfg->profile_log_size);
#endif
        if (ucc_global_config.log_component.log_level >= UCC_LOG_LEVEL_INFO) {
            ret = dladdr(ucc_init_version, &dl_info);
            if (ret == 0) {
                ucc_error("failed to get ucc_init_version handler");
                return UCC_ERR_NO_MESSAGE;
            }
            ucc_info("version: %s, loaded from: %s, cfg file: %s",
                     ucc_get_version_string(), dl_info.dli_fname,
                     ucc_global_config.file_cfg ?
                     ucc_global_config.file_cfg->filename: "n/a");
        }
    }
    return UCC_OK;
}

__attribute__((destructor)) static void ucc_destructor(void)
{
    if (ucc_global_config.initialized) {
#ifdef HAVE_PROFILING
        ucc_profile_cleanup();
#endif
        ucc_config_parser_release_opts(&ucc_global_config,
                                       ucc_global_config_table);
        if (ucc_global_config.file_cfg) {
            ucc_release_file_config(ucc_global_config.file_cfg);
        }
        free(ucc_global_config.component_path);
        free(ucc_global_config.install_path);
    }
}
